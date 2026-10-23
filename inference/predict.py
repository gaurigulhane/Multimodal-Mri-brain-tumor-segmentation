"""
predict.py
----------
Inference pipeline for multimodal MRI brain tumor segmentation.

Loads a trained checkpoint (or runs with random weights if none provided),
preprocesses four NIfTI MRI modalities, and returns per-region binary masks
together with probability maps.

Usage
-----
  # Predict from command line
  python inference/predict.py \\
      --checkpoint checkpoints/best.pth \\
      --t1      patient/t1.nii.gz \\
      --t1ce    patient/t1ce.nii.gz \\
      --t2      patient/t2.nii.gz \\
      --flair   patient/flair.nii.gz \\
      --output  outputs/prediction.nii.gz
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional nibabel import
# ---------------------------------------------------------------------------
try:
    import nibabel as nib
    _NIBABEL_AVAILABLE = True
except ImportError:
    _NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not installed – NIfTI loading disabled.", RuntimeWarning)

# ---------------------------------------------------------------------------
# Optional models import
# ---------------------------------------------------------------------------
try:
    from models import get_model  # type: ignore
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===========================================================================
# Fallback model
# ===========================================================================

class _FallbackModel(nn.Module):
    """Minimal 3-D conv model used when the real model module is absent."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# TumorPredictor
# ===========================================================================

class TumorPredictor:
    """End-to-end inference pipeline for brain tumor segmentation.

    Parameters
    ----------
    checkpoint_path : str | None
        Path to a trained .pth checkpoint.  If None or the file does not exist
        the model is initialised with random weights and a warning is issued.
    model_type : str
        Architecture selector passed to ``get_model``.  'improved' or 'baseline'.
    device : str
        'auto' (default) selects CUDA if available, else CPU.
        Can also be 'cuda', 'cpu', 'cuda:0', etc.
    """

    # Default architecture params (match config.yaml)
    _IN_CHANNELS: int = 4
    _OUT_CHANNELS: int = 3
    _FEATURES: list = [32, 64, 128, 256]
    _BOTTLENECK: int = 512

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "improved",
        device: str = "auto",
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self._model_loaded = False
        self._checkpoint_used: Optional[str] = None
        self._warning: Optional[str] = None

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("TumorPredictor device: %s", self.device)

        # Build & load model
        self.model: nn.Module = self._load_model()
        self.model.eval()

    # -----------------------------------------------------------------------
    # Model initialisation
    # -----------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        """Instantiate the model and load weights from checkpoint if available.

        Returns
        -------
        nn.Module on self.device, in eval mode.
        """
        # Check weights first to determine architecture dimensions if present
        ckpt_path = Path(self.checkpoint_path) if self.checkpoint_path else None
        features = self._FEATURES
        bottleneck = self._BOTTLENECK
        state_dict = None
        loaded_successfully = False

        if ckpt_path is not None and ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                if "features" in ckpt:
                    features = ckpt["features"]
                    bottleneck = ckpt.get("bottleneck_features", bottleneck)
                elif isinstance(state_dict, dict) and "encoder_blocks.0.res_block.conv1.weight" in state_dict:
                    f0 = state_dict["encoder_blocks.0.res_block.conv1.weight"].shape[0]
                    if f0 == 16:
                        features = [16, 32, 64, 128]
                        bottleneck = 256
                    elif f0 == 32:
                        features = [32, 64, 128, 256]
                        bottleneck = 512
                loaded_successfully = True
            except Exception as exc:
                logger.warning(f"Failed to inspect checkpoint '{ckpt_path}': {exc}")

        # Instantiate architecture
        if _MODELS_AVAILABLE:
            model = get_model(
                model_type=self.model_type,
                in_channels=self._IN_CHANNELS,
                out_channels=self._OUT_CHANNELS,
                features=features,
                bottleneck_features=bottleneck,
            )
        else:
            logger.warning("models module not found - using fallback minimal model.")
            model = _FallbackModel(
                in_channels=self._IN_CHANNELS,
                out_channels=self._OUT_CHANNELS,
            )

        # Load weights if available
        if loaded_successfully and state_dict is not None:
            try:
                model.load_state_dict(state_dict, strict=True)
                self._model_loaded = True
                self._checkpoint_used = str(ckpt_path)
                logger.info("Loaded checkpoint: %s", ckpt_path)
            except Exception as exc:
                self._warning = (
                    f"Failed to load checkpoint '{ckpt_path}': {exc}. "
                    "Running with random weights."
                )
                logger.warning(self._warning)
        else:
            self._warning = (
                f"No checkpoint provided or file not found "
                f"('{self.checkpoint_path}'). Running with random weights."
            )
            logger.warning(self._warning)

        model.to(self.device)
        return model

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_nifti(path: str) -> np.ndarray:
        """Load a NIfTI file and return a float32 numpy array.

        Parameters
        ----------
        path : str  Path to the .nii or .nii.gz file.

        Returns
        -------
        np.ndarray (D, H, W) float32.

        Raises
        ------
        ImportError  if nibabel is not installed.
        FileNotFoundError  if the file does not exist.
        """
        if not _NIBABEL_AVAILABLE:
            raise ImportError(
                "nibabel is required to load NIfTI files: pip install nibabel"
            )
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NIfTI file not found: '{path}'")
        img = nib.load(str(p))
        data = np.asarray(img.dataobj, dtype=np.float32)
        return data

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        """Z-score normalise a 3-D volume using non-zero voxel statistics.

        Parameters
        ----------
        volume : np.ndarray (D, H, W)

        Returns
        -------
        Normalised float32 array.
        """
        mask = volume > 0
        if mask.sum() == 0:
            return volume.astype(np.float32)
        mean = volume[mask].mean()
        std = volume[mask].std()
        if std < 1e-8:
            std = 1.0
        out = np.zeros_like(volume, dtype=np.float32)
        out[mask] = (volume[mask] - mean) / std
        return out

    def preprocess(
        self,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        flair_path: str,
    ) -> torch.Tensor:
        """Load and preprocess four MRI modalities into a model-ready tensor.

        Parameters
        ----------
        t1_path    : str  Path to T1 NIfTI.
        t1ce_path  : str  Path to T1-contrast-enhanced NIfTI.
        t2_path    : str  Path to T2 NIfTI.
        flair_path : str  Path to FLAIR NIfTI.

        Returns
        -------
        torch.Tensor shape (1, 4, D, H, W) on self.device.
        """
        vols = []
        for path, name in [
            (t1_path, "T1"),
            (t1ce_path, "T1ce"),
            (t2_path, "T2"),
            (flair_path, "FLAIR"),
        ]:
            vol = self._load_nifti(path)
            vol = self._normalize(vol)
            vols.append(vol)
            logger.debug("Loaded %s: shape=%s", name, vol.shape)

        stacked = np.stack(vols, axis=0)  # (4, D, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, 4, D, H, W)
        return tensor.to(self.device)

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        flair_path: str,
    ) -> dict:
        """Run end-to-end inference from NIfTI file paths.

        Parameters
        ----------
        t1_path, t1ce_path, t2_path, flair_path : str
            Paths to the four MRI modality NIfTI files.

        Returns
        -------
        dict:
        {
          'prediction'           : np.ndarray (3, D, H, W) uint8 – binary masks,
          'probability'          : np.ndarray (3, D, H, W) float32 – sigmoid probs,
          'regions': {
            'WT': np.ndarray (D, H, W) uint8,
            'TC': np.ndarray (D, H, W) uint8,
            'ET': np.ndarray (D, H, W) uint8,
          },
          'inference_time_seconds': float,
          'model_loaded'         : bool,
          'checkpoint_used'      : str | None,
          'warning'              : str | None,
        }
        """
        tensor = self.preprocess(t1_path, t1ce_path, t2_path, flair_path)
        return self._run_inference(tensor)

    def predict_from_arrays(
        self,
        t1: np.ndarray,
        t1ce: np.ndarray,
        t2: np.ndarray,
        flair: np.ndarray,
    ) -> dict:
        """Run inference directly from pre-loaded numpy arrays.

        Parameters
        ----------
        t1, t1ce, t2, flair : np.ndarray (D, H, W)
            Pre-loaded (and optionally pre-normalised) MRI volumes.

        Returns
        -------
        Same dict as :meth:`predict`.
        """
        vols = []
        for vol, name in [(t1, "T1"), (t1ce, "T1ce"), (t2, "T2"), (flair, "FLAIR")]:
            vol_f = vol.astype(np.float32)
            norm = self._normalize(vol_f)
            vols.append(norm)
            logger.debug("Array %s: shape=%s", name, norm.shape)

        stacked = np.stack(vols, axis=0)  # (4, D, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        return self._run_inference(tensor)

    def _run_inference(self, tensor: torch.Tensor) -> dict:
        """Internal inference runner.

        Parameters
        ----------
        tensor : torch.Tensor (1, 4, D, H, W) on self.device.

        Returns
        -------
        Prediction result dict.
        """
        t0 = time.perf_counter()

        with torch.no_grad():
            out = self.model(tensor)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits)         # (1, 3, D, H, W)

        inference_time = time.perf_counter() - t0

        prob_np = probs[0].cpu().numpy().astype(np.float32)  # (3, D, H, W)
        pred_np = (prob_np > 0.5).astype(np.uint8)           # (3, D, H, W)

        result = {
            "prediction": pred_np,
            "probability": prob_np,
            "regions": {
                "WT": pred_np[0],
                "TC": pred_np[1],
                "ET": pred_np[2],
            },
            "inference_time_seconds": round(inference_time, 4),
            "model_loaded": self._model_loaded,
            "checkpoint_used": self._checkpoint_used,
            "warning": self._warning,
        }

        logger.info(
            "Inference complete in %.3f s | WT=%d TC=%d ET=%d voxels",
            inference_time,
            int(pred_np[0].sum()),
            int(pred_np[1].sum()),
            int(pred_np[2].sum()),
        )
        return result

    # -----------------------------------------------------------------------
    # Model info
    # -----------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return metadata about the loaded model.

        Returns
        -------
        dict:
        {
          'model_name': str,
          'model_type': str,
          'n_parameters': int,
          'n_trainable_parameters': int,
          'device': str,
          'model_loaded': bool,
          'checkpoint_used': str | None,
        }
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": type(self.model).__name__,
            "model_type": self.model_type,
            "n_parameters": n_params,
            "n_trainable_parameters": n_trainable,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "checkpoint_used": self._checkpoint_used,
        }


# ===========================================================================
# CLI helpers
# ===========================================================================

def _save_prediction_nifti(
    pred_np: np.ndarray,
    reference_nifti_path: str,
    output_path: str,
) -> None:
    """Save a prediction array as a NIfTI file, reusing the reference affine.

    Parameters
    ----------
    pred_np              : np.ndarray (3, D, H, W) or (D, H, W)
    reference_nifti_path : str  Path to any input NIfTI (for affine/header).
    output_path          : str  Destination .nii.gz path.
    """
    if not _NIBABEL_AVAILABLE:
        logger.warning("nibabel not available – skipping NIfTI save.")
        return

    ref = nib.load(reference_nifti_path)
    affine = ref.affine
    header = ref.header

    # Combine 3-channel prediction to a single label map for saving:
    # ET=4, TC=1 (non-ET part), ED=2 (WT but not TC)
    if pred_np.ndim == 4:
        wt = pred_np[0].astype(bool)
        tc = pred_np[1].astype(bool)
        et = pred_np[2].astype(bool)
        label_map = np.zeros(wt.shape, dtype=np.uint8)
        label_map[wt] = 2          # oedema / peritumoral
        label_map[tc] = 1          # tumour core
        label_map[et] = 4          # enhancing tumour
    else:
        label_map = pred_np.astype(np.uint8)

    out_img = nib.Nifti1Image(label_map, affine=affine, header=header)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    logger.info("Prediction saved → %s", out_path)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Brain tumor segmentation inference pipeline."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained checkpoint (.pth). Omit to run with random weights.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="improved",
        choices=["improved", "baseline"],
        help="Model architecture type.",
    )
    parser.add_argument(
        "--t1",
        type=str,
        required=True,
        help="Path to T1 NIfTI file.",
    )
    parser.add_argument(
        "--t1ce",
        type=str,
        required=True,
        help="Path to T1-contrast-enhanced NIfTI file.",
    )
    parser.add_argument(
        "--t2",
        type=str,
        required=True,
        help="Path to T2 NIfTI file.",
    )
    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Path to FLAIR NIfTI file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/prediction.nii.gz",
        help="Output path for the predicted segmentation NIfTI.",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="Optional path to save inference metadata as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu', 'cuda:0', etc.",
    )
    args = parser.parse_args()

    predictor = TumorPredictor(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
    )

    # Print model info
    info = predictor.get_model_info()
    logger.info("Model info: %s", json.dumps(info, indent=2))

    # Run prediction
    result = predictor.predict(
        t1_path=args.t1,
        t1ce_path=args.t1ce,
        t2_path=args.t2,
        flair_path=args.flair,
    )

    # Save NIfTI output
    _save_prediction_nifti(result["prediction"], args.t1, args.output)

    # Save JSON metadata
    if args.json_output:
        meta = {
            "inference_time_seconds": result["inference_time_seconds"],
            "model_loaded": result["model_loaded"],
            "checkpoint_used": result["checkpoint_used"],
            "warning": result["warning"],
            "voxel_counts": {
                "WT": int(result["regions"]["WT"].sum()),
                "TC": int(result["regions"]["TC"].sum()),
                "ET": int(result["regions"]["ET"].sum()),
            },
            "model_info": info,
        }
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Inference metadata saved → %s", json_path)

    if result.get("warning"):
        print(f"\n[WARNING] {result['warning']}")

    print(
        f"\nDone! Inference time: {result['inference_time_seconds']:.3f}s  "
        f"| Tumour voxels → WT={result['regions']['WT'].sum()}  "
        f"TC={result['regions']['TC'].sum()}  "
        f"ET={result['regions']['ET'].sum()}"
    )


if __name__ == "__main__":
    main()
