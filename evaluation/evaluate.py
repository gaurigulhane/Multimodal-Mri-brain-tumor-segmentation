"""
evaluate.py
-----------
Standalone evaluation script for trained brain tumor segmentation models.

Usage
-----
  python evaluation/evaluate.py \\
      --checkpoint checkpoints/best.pth \\
      --dataset brats2021 \\
      --data_root ./data/BraTS2021 \\
      --output_dir outputs/evaluation
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from models import get_model  # type: ignore
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

try:
    from datasets import BraTS2021DataModule, BraTS2023METDataModule  # type: ignore
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

from evaluation.metrics import SegmentationMetrics, MetricsAccumulator, compute_hd95

logger = logging.getLogger(__name__)


# ===========================================================================
# Logging helper
# ===========================================================================

def _setup_logging(log_dir: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ===========================================================================
# Fallback model for when models module is absent
# ===========================================================================

class _FallbackModel(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# Evaluator
# ===========================================================================

class Evaluator:
    """Loads a trained model and evaluates it on a BraTS dataset.

    Parameters
    ----------
    model_checkpoint : str
        Path to the .pth checkpoint file produced by Trainer.
    dataset          : str
        Dataset name: 'brats2021' or 'brats2023_met'.
    data_root        : str
        Root directory of the dataset.
    config_path      : str
        Path to config.yaml (used for model architecture).
    output_dir       : str
        Directory where reports are written.
    """

    def __init__(
        self,
        model_checkpoint: str,
        dataset: str = "brats2021",
        data_root: str = "",
        config_path: str = "training/config.yaml",
        output_dir: str = "outputs/evaluation",
    ):
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset.lower()
        self.data_root = data_root
        self.config_path = config_path
        self.output_dir = Path(output_dir)

        _setup_logging(str(self.output_dir / "logs"))

        # Load config if available
        self.cfg = self._load_config()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Evaluation device: %s", self.device)

        # Model (loaded in load_model)
        self.model: nn.Module = None

        # Metrics engine
        self.metrics_engine = SegmentationMetrics(threshold=0.5)

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------

    def _load_config(self) -> dict:
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        logger.warning("Config not found at '%s', using defaults.", self.config_path)
        return {
            "model": {
                "type": "improved",
                "in_channels": 4,
                "out_channels": 3,
                "features": [32, 64, 128, 256],
                "bottleneck_features": 512,
            }
        }

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model architecture and weights from the checkpoint."""
        model_cfg = self.cfg.get("model", {})
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
        else:
            logger.warning("models module unavailable – using fallback model.")
            self.model = _FallbackModel(in_channels=in_ch, out_channels=out_ch)

        # Load weights
        ckpt_path = Path(self.model_checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded weights from '%s'", ckpt_path)
        else:
            logger.warning("Checkpoint '%s' not found – evaluating with random weights!", ckpt_path)

        self.model.to(self.device)
        self.model.eval()

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------

    def evaluate_dataset(
        self, loader: DataLoader, dataset_name: str = "brats2021"
    ) -> dict:
        """Evaluate the model on all batches in *loader*.

        Parameters
        ----------
        loader       : DataLoader  Validation / test loader.
        dataset_name : str         Used only for labelling the output dict.

        Returns
        -------
        dict with:
          - 'per_patient': list of per-batch metric dicts
          - 'mean': aggregated mean metrics across all batches
          - 'dataset': dataset_name
          - 'n_samples': int
        """
        accumulator = MetricsAccumulator()
        per_patient: list = []
        n_samples = 0

        logger.info("Starting evaluation on '%s' (%d batches)...", dataset_name, len(loader))

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    labels = batch["label"].to(self.device, non_blocking=True)
                    patient_ids = batch.get("patient_id", [f"sample_{batch_idx}"])
                else:
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    patient_ids = [f"sample_{batch_idx + i}" for i in range(images.shape[0])]

                out = self.model(images)
                logits = out["logits"] if isinstance(out, dict) else out
                metrics = self.metrics_engine.compute(logits, labels)

                # Record per-patient
                B = images.shape[0]
                n_samples += B
                for i in range(B):
                    pid = patient_ids[i] if i < len(patient_ids) else f"sample_{batch_idx}_{i}"
                    per_patient.append({"patient_id": pid, **metrics})

                accumulator.update(metrics)

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    logger.info(
                        "  Batch [%d/%d]  mean_dice=%.4f",
                        batch_idx + 1,
                        len(loader),
                        metrics.get("mean_dice", 0.0),
                    )

        mean_metrics = accumulator.compute_mean()
        logger.info(
            "Evaluation complete: n_samples=%d  mean_dice=%.4f",
            n_samples,
            mean_metrics.get("mean_dice", 0.0),
        )

        return {
            "dataset": dataset_name,
            "n_samples": n_samples,
            "mean": mean_metrics,
            "per_patient": per_patient,
        }

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def generate_report(self, results: dict, output_dir: str) -> None:
        """Save a JSON result file and a Markdown summary report.

        Parameters
        ----------
        results    : dict  Return value of :meth:`evaluate_dataset`.
        output_dir : str   Directory where files are saved.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # --- JSON report ---
        json_path = out_path / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("JSON report saved → %s", json_path)

        # --- Markdown summary ---
        mean = results.get("mean", {})
        md_lines = [
            "# Brain Tumor Segmentation – Evaluation Report",
            "",
            f"**Dataset**: {results.get('dataset', 'N/A')}  ",
            f"**Samples evaluated**: {results.get('n_samples', 0)}  ",
            f"**Checkpoint**: `{self.model_checkpoint}`  ",
            "",
            "## Mean Metrics",
            "",
            "| Region | Dice | IoU | Precision | Recall | Specificity |",
            "|--------|------|-----|-----------|--------|-------------|",
        ]

        for region in ["WT", "TC", "ET"]:
            prefix = f"{region}/"
            dice = mean.get(f"{prefix}dice", mean.get(f"WT/dice" if region == "WT" else "", 0.0))
            # Try both flat and nested key patterns
            def _get(key_nested: str, key_flat: str) -> str:
                v = mean.get(key_nested, mean.get(key_flat, None))
                return f"{v:.4f}" if v is not None else "N/A"

            dice_v = _get(f"{region}/dice", "dice")
            iou_v = _get(f"{region}/iou", "iou")
            prec_v = _get(f"{region}/precision", "precision")
            rec_v = _get(f"{region}/recall", "recall")
            spec_v = _get(f"{region}/specificity", "specificity")
            md_lines.append(
                f"| {region} | {dice_v} | {iou_v} | {prec_v} | {rec_v} | {spec_v} |"
            )

        mean_dice = mean.get("mean_dice", 0.0)
        md_lines += [
            "",
            f"**Mean Dice (WT+TC+ET)**: `{mean_dice:.4f}`",
            "",
            "## Per-Patient Summary",
            "",
            f"Full per-patient data saved to `evaluation_results.json`.",
            "",
        ]

        md_path = out_path / "evaluation_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info("Markdown report saved → %s", md_path)

    # -----------------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Load model, evaluate, and generate report.

        Returns
        -------
        Evaluation result dict.
        """
        self.load_model()

        if not _DATASETS_AVAILABLE:
            raise RuntimeError(
                "datasets module not found. Cannot create data loaders. "
                "Install or fix the 'datasets' package and re-run."
            )

        ds_cfg = self.cfg.get("dataset", {})
        patch_size = tuple(ds_cfg.get("patch_size", [128, 128, 128]))
        batch_size = self.cfg.get("training", {}).get("batch_size", 1)
        num_workers = ds_cfg.get("num_workers", 0)

        data_root = self.data_root or ds_cfg.get("data_root", "./data/BraTS2021")

        if self.dataset_name == "brats2021":
            dm = BraTS2021DataModule(
                data_root=data_root,
                patch_size=patch_size,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        elif self.dataset_name in ("brats2023_met", "brats2023"):
            dm = BraTS2023METDataModule(
                data_root=data_root,
                patch_size=patch_size,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'")

        dm.setup()
        test_loader = dm.test_dataloader()

        results = self.evaluate_dataset(test_loader, dataset_name=self.dataset_name)
        self.generate_report(results, str(self.output_dir))
        return results


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained brain tumor segmentation model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="brats2021",
        choices=["brats2021", "brats2023_met"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Root directory of the dataset (overrides config if provided).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to config.yaml for model architecture.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation outputs.",
    )
    args = parser.parse_args()

    evaluator = Evaluator(
        model_checkpoint=args.checkpoint,
        dataset=args.dataset,
        data_root=args.data_root,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
