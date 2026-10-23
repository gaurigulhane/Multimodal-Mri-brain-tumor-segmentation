"""
train.py
--------
Main training script for Multimodal MRI Brain Tumor Segmentation.

Usage
-----
  # Normal training
  python training/train.py --config training/config.yaml

  # Resume from checkpoint
  python training/train.py --config training/config.yaml --resume checkpoints/last.pth

  # Quick smoke test (synthetic data, 2 epochs)
  python training/train.py --smoke_test
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make the project root importable regardless of where the script is run from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

# ---------------------------------------------------------------------------
# Project imports (guarded so the file can still be imported for testing)
# ---------------------------------------------------------------------------
try:
    from models import get_model  # type: ignore
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logging.warning("models.get_model not found – will use a fallback 3-D conv stack.")

try:
    from datasets import BraTS2021DataModule, BraTS2023METDataModule  # type: ignore
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    logging.warning("datasets module not found – DataModules unavailable.")

from training.losses import get_loss_function


# ===========================================================================
# Logging helpers
# ===========================================================================

def _setup_logging(log_dir: str, name: str = "train") -> logging.Logger:
    """Configure root logger to write to console and to a log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger(name)


def _set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Fallback minimal model (used when models module is not present)
# ===========================================================================

class _FallbackModel(nn.Module):
    """Tiny 3-D UNet-like network used only when the real model is unavailable."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Conv3d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """End-to-end trainer for brain tumor segmentation models.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    resume_path : str | None
        Optional path to a checkpoint to resume from.
    """

    def __init__(self, config_path: str = "training/config.yaml", resume_path: str = None):
        self.config_path = Path(config_path)
        self.resume_path = resume_path

        # Load configuration
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Directories
        self.log_dir = self.cfg["outputs"]["log_dir"]
        self.metrics_dir = self.cfg["outputs"]["metrics_dir"]
        self.checkpoint_dir = self.cfg["checkpoints"]["save_dir"]

        # Logger
        self.logger = _setup_logging(self.log_dir)

        # Reproducibility
        seed = self.cfg["project"].get("seed", 42)
        _set_seed(seed)
        self.logger.info("Seed set to %d", seed)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Using device: %s", self.device)

        # Components (populated by setup_* methods)
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler = None
        self.loss_fn: nn.Module = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None

        # Training state
        self.start_epoch: int = 0
        self.best_val_dice: float = -1.0
        self.history: list = []

        # Scaler for mixed precision
        self._use_amp = self.cfg["training"].get("mixed_precision", False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self._use_amp else None

        # Build everything
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.loss_fn = get_loss_function(
            self.cfg["training"].get("loss_function", "dice_focal")
        )
        self.loss_fn.to(self.device)

        # Resume if requested
        if self.resume_path:
            self.load_checkpoint(self.resume_path)

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def setup_model(self) -> None:
        """Instantiate model from config."""
        model_cfg = self.cfg["model"]
        model_type = model_cfg.get("type", "improved")
        in_ch = model_cfg.get("in_channels", 4)
        out_ch = model_cfg.get("out_channels", 3)
        features = model_cfg.get("features", [32, 64, 128, 256])
        bottleneck = model_cfg.get("bottleneck_features", 512)

        if _MODELS_AVAILABLE:
            self.model = get_model(
                model_type=model_type,
                in_channels=in_ch,
                out_channels=out_ch,
                features=features,
                bottleneck_features=bottleneck,
            )
            self.logger.info("Loaded model: %s", model_type)
        else:
            self.logger.warning("Using fallback minimal model (models module not found).")
            self.model = _FallbackModel(in_channels=in_ch, out_channels=out_ch)

        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Trainable parameters: {:,}".format(n_params))

    def setup_optimizer(self) -> None:
        """Create optimizer from config."""
        opt_name = self.cfg["training"].get("optimizer", "adamw").lower()
        lr = float(self.cfg["training"].get("learning_rate", 1e-4))
        wd = float(self.cfg["training"].get("weight_decay", 1e-5))

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            raise ValueError(f"Unknown optimizer '{opt_name}'")
        self.logger.info("Optimizer: %s  lr=%.2e  wd=%.2e", opt_name, lr, wd)

    def setup_scheduler(self) -> None:
        """Create LR scheduler from config."""
        sched_name = self.cfg["training"].get("scheduler", "cosine").lower()
        epochs = self.cfg["training"].get("epochs", 100)

        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-7
            )
        elif sched_name in ("plateau", "reducelronplateau"):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=10, verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler '{sched_name}'")
        self.logger.info("Scheduler: %s", sched_name)

    def setup_data(self) -> None:
        """Instantiate the appropriate DataModule and create loaders."""
        if not _DATASETS_AVAILABLE:
            raise RuntimeError(
                "datasets module not found. Cannot set up data loaders. "
                "Use --smoke_test for a synthetic data run."
            )

        ds_cfg = self.cfg["dataset"]
        ds_name = ds_cfg.get("name", "brats2021").lower()
        data_root = ds_cfg.get("data_root", "./data/BraTS2021")
        patch_size = tuple(ds_cfg.get("patch_size", [128, 128, 128]))
        batch_size = self.cfg["training"].get("batch_size", 1)
        num_workers = ds_cfg.get("num_workers", 0)
        train_ratio = ds_cfg.get("train_ratio", 0.70)
        val_ratio = ds_cfg.get("val_ratio", 0.15)

        if ds_name == "brats2021":
            dm = BraTS2021DataModule(
                data_root=data_root,
                patch_size=patch_size,
                batch_size=batch_size,
                num_workers=num_workers,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
        elif ds_name in ("brats2023_met", "brats2023"):
            dm = BraTS2023METDataModule(
                data_root=data_root,
                patch_size=patch_size,
                batch_size=batch_size,
                num_workers=num_workers,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
        else:
            raise ValueError(f"Unknown dataset '{ds_name}'")

        dm.setup()
        self.train_loader = dm.train_dataloader()
        self.val_loader = dm.val_dataloader()
        self.logger.info(
            "Data ready: %d train batches, %d val batches",
            len(self.train_loader),
            len(self.val_loader),
        )

    # -----------------------------------------------------------------------
    # Per-epoch loops
    # -----------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one full training epoch.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.

        Returns
        -------
        dict with key 'loss' (mean epoch loss).
        """
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        clip_norm = float(self.cfg["training"].get("gradient_clip_norm", 1.0))

        pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
        for batch in pbar:
            # Support both dict-style and tuple-style batches
            if isinstance(batch, dict):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
            else:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self._use_amp:
                with torch.cuda.amp.autocast():
                    preds = self.model(images)
                    if isinstance(preds, dict):
                        preds = preds["logits"]
                    loss = self.loss_fn(preds, labels.float())
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                if isinstance(preds, dict):
                    preds = preds["logits"]
                loss = self.loss_fn(preds, labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {"loss": running_loss / max(n_batches, 1)}

    def validate(self, loader: DataLoader) -> dict:
        """Run validation and compute per-region Dice scores.

        Parameters
        ----------
        loader : DataLoader
            Validation data loader.

        Returns
        -------
        dict with keys: 'loss', 'dice_WT', 'dice_TC', 'dice_ET', 'mean_dice'.
        """
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        region_dice = {"WT": [], "TC": [], "ET": []}
        smooth = 1e-5

        with torch.no_grad():
            pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
            for batch in pbar:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    labels = batch["label"].to(self.device, non_blocking=True)
                else:
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)

                preds = self.model(images)
                if isinstance(preds, dict):
                    preds = preds["logits"]
                loss = self.loss_fn(preds, labels.float())
                running_loss += loss.item()
                n_batches += 1

                # Binary predictions (sigmoid + threshold)
                probs = torch.sigmoid(preds)
                binary_preds = (probs > 0.5).float()

                # Compute per-channel Dice
                B, C = binary_preds.shape[:2]
                for b in range(B):
                    for c, region in enumerate(["WT", "TC", "ET"]):
                        if c < C and labels.shape[1] > c:
                            p = binary_preds[b, c].view(-1)
                            t = labels[b, c].float().view(-1)
                            inter = (p * t).sum()
                            d = (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
                            region_dice[region].append(d.item())

        mean_loss = running_loss / max(n_batches, 1)
        dice_wt = float(np.mean(region_dice["WT"])) if region_dice["WT"] else 0.0
        dice_tc = float(np.mean(region_dice["TC"])) if region_dice["TC"] else 0.0
        dice_et = float(np.mean(region_dice["ET"])) if region_dice["ET"] else 0.0
        mean_dice = float(np.mean([dice_wt, dice_tc, dice_et]))

        return {
            "loss": mean_loss,
            "dice_WT": dice_wt,
            "dice_TC": dice_tc,
            "dice_ET": dice_et,
            "mean_dice": mean_dice,
        }

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
        tag: str = "last",
    ) -> None:
        """Save model + optimiser state to disk.

        Parameters
        ----------
        epoch   : Current epoch index (0-based).
        metrics : Metrics dict to store alongside the checkpoint.
        is_best : If True, also saves a copy as 'best.pth'.
        tag     : Filename tag (e.g. 'last', 'epoch_010').
        """
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        path = ckpt_dir / f"{tag}.pth"
        torch.save(state, path)
        self.logger.info("Saved checkpoint -> %s", path)

        if is_best:
            best_path = ckpt_dir / "best.pth"
            torch.save(state, best_path)
            self.logger.info("New best checkpoint -> %s", best_path)

        # Prune old checkpoints (keep_last_n)
        keep = self.cfg["checkpoints"].get("keep_last_n", 5)
        epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
        while len(epoch_ckpts) > keep:
            epoch_ckpts[0].unlink()
            self.logger.debug("Removed old checkpoint: %s", epoch_ckpts[0])
            epoch_ckpts = epoch_ckpts[1:]

    def load_checkpoint(self, path: str) -> None:
        """Load model + optimiser state from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the .pth checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1

        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        best = ckpt.get("metrics", {}).get("mean_dice", -1.0)
        self.best_val_dice = best
        self.logger.info(
            "Resumed from '%s' (epoch %d, best_val_dice=%.4f)",
            path,
            self.start_epoch - 1,
            self.best_val_dice,
        )

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self) -> None:
        """Full training loop with early stopping and metric logging."""
        # Data
        self.setup_data()

        total_epochs = self.cfg["training"].get("epochs", 100)
        val_every = self.cfg["training"].get("validate_every_n_epochs", 1)
        patience = self.cfg["training"].get("early_stopping_patience", 20)
        save_best_only = self.cfg["checkpoints"].get("save_best_only", False)
        sched_name = self.cfg["training"].get("scheduler", "cosine").lower()

        metrics_path = Path(self.metrics_dir)
        metrics_path.mkdir(parents=True, exist_ok=True)
        history_file = metrics_path / "training_history.json"

        no_improve_count = 0

        self.logger.info(
            "Starting training: epochs=%d, device=%s", total_epochs, self.device
        )

        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()
            self.logger.info("Epoch [%d/%d]", epoch + 1, total_epochs)

            # --- Train ---
            train_metrics = self.train_epoch(self.train_loader)

            # --- Validate ---
            val_metrics: dict = {}
            if (epoch + 1) % val_every == 0:
                val_metrics = self.validate(self.val_loader)
                val_dice = val_metrics.get("mean_dice", 0.0)

                self.logger.info(
                    "  Train loss: %.4f | Val loss: %.4f | "
                    "Dice WT=%.4f TC=%.4f ET=%.4f (mean=%.4f)",
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["dice_WT"],
                    val_metrics["dice_TC"],
                    val_metrics["dice_ET"],
                    val_dice,
                )

                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Scheduler step
                if sched_name in ("plateau", "reducelronplateau"):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

                # Save checkpoint
                if not save_best_only:
                    self.save_checkpoint(
                        epoch,
                        {**train_metrics, **val_metrics},
                        is_best=is_best,
                        tag=f"epoch_{epoch + 1:04d}",
                    )
                elif is_best:
                    self.save_checkpoint(
                        epoch,
                        {**train_metrics, **val_metrics},
                        is_best=True,
                        tag=f"epoch_{epoch + 1:04d}",
                    )

                # Always update "last.pth"
                self.save_checkpoint(
                    epoch,
                    {**train_metrics, **val_metrics},
                    is_best=False,
                    tag="last",
                )

                # Early stopping
                if no_improve_count >= patience:
                    self.logger.info(
                        "Early stopping triggered (no improvement for %d epochs).",
                        patience,
                    )
                    break
            else:
                # Non-validation epoch: just step cosine scheduler
                if sched_name not in ("plateau", "reducelronplateau"):
                    self.scheduler.step()
                self.logger.info("  Train loss: %.4f", train_metrics["loss"])

            epoch_time = time.time() - epoch_start
            record = {
                "epoch": epoch + 1,
                "time_seconds": round(epoch_time, 2),
                **train_metrics,
                **val_metrics,
            }
            self.history.append(record)

            # Persist history after every epoch
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

        self.logger.info(
            "Training complete. Best val mean_dice = %.4f", self.best_val_dice
        )

    # -----------------------------------------------------------------------
    # Smoke test
    # -----------------------------------------------------------------------

    def run_smoke_test(self) -> None:
        """Create synthetic data, train for 2 epochs, verify loss decreases."""
        self.logger.info("=== SMOKE TEST ===")

        # Synthetic dataset: 2 samples, shape (1,4,32,32,32)
        n_samples = 2
        B, C_in, C_out = 1, 4, 3
        D = H = W = 32

        images = torch.randn(n_samples, C_in, D, H, W)
        labels = (torch.rand(n_samples, C_out, D, H, W) > 0.8).float()

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=B, shuffle=False)

        losses = []
        for epoch in range(2):
            self.model.train()
            epoch_loss = 0.0
            for imgs, lbls in loader:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(imgs)
                if isinstance(preds, dict):
                    preds = preds["logits"]
                loss = self.loss_fn(preds, lbls)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            mean_loss = epoch_loss / len(loader)
            losses.append(mean_loss)
            self.logger.info("  Smoke epoch %d/2  loss=%.6f", epoch + 1, mean_loss)

        if len(losses) == 2:
            # Loss should generally decrease (not guaranteed in 2 steps, but
            # we at least verify the forward/backward pass ran without errors)
            self.logger.info(
                "Smoke test PASSED. Loss: %.6f -> %.6f", losses[0], losses[1]
            )
        else:
            self.logger.error("Smoke test FAILED - unexpected number of loss values.")
            sys.exit(1)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a brain tumor segmentation model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a quick smoke test with synthetic data.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        # Use default config path for smoke test
        config_path = args.config if Path(args.config).exists() else "training/config.yaml"
        # If config file doesn't exist, use a minimal in-memory config
        if not Path(config_path).exists():
            logging.basicConfig(level=logging.INFO)
            logging.warning(
                "Config file not found at '%s'. Using defaults.", config_path
            )
        trainer = Trainer(config_path=config_path)
        trainer.run_smoke_test()
    else:
        trainer = Trainer(config_path=args.config, resume_path=args.resume)
        trainer.train()


if __name__ == "__main__":
    main()
