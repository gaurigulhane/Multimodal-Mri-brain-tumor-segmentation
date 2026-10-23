"""
metrics.py
----------
Segmentation metrics for multimodal MRI brain tumor segmentation.

Supports:
  - compute_dice          : Soft/binary Dice coefficient
  - compute_iou           : Intersection-over-Union (Jaccard)
  - compute_precision     : Precision (PPV)
  - compute_recall        : Recall (Sensitivity / TPR)
  - compute_specificity   : Specificity (TNR)
  - compute_hd95          : 95th-percentile Hausdorff distance
  - SegmentationMetrics   : End-to-end metric computation for (B,3,D,H,W) tensors
  - MetricsAccumulator    : Running accumulator for multi-batch averaging
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scipy import for HD95
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import distance_transform_edt  # noqa: F401
    from scipy.spatial.distance import directed_hausdorff  # noqa: F401
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not available – compute_hd95 will return -1.0")


# ===========================================================================
# BraTS label mapping
# ===========================================================================
# Raw BraTS labels:
#   0 = background
#   1 = necrotic / non-enhancing tumour core (NCR/NET)
#   2 = peritumoral oedema (ED)
#   4 = GD-enhancing tumour (ET)   [note: label 3 is not used]
#
# Hierarchical tumour regions:
#   WT (Whole Tumour)          = labels 1 + 2 + 4
#   TC (Tumour Core)           = labels 1 + 4
#   ET (Enhancing Tumour)      = label  4

BRATS_LABEL_MAP = {
    "WT": [1, 2, 4],   # channel 0
    "TC": [1, 4],       # channel 1
    "ET": [4],          # channel 2
}


def _labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """Convert a BraTS integer-label volume to a 3-channel binary array.

    Parameters
    ----------
    seg : np.ndarray shape (D, H, W) with values in {0, 1, 2, 4}

    Returns
    -------
    np.ndarray shape (3, D, H, W) – channels: WT, TC, ET
    """
    wt = np.isin(seg, [1, 2, 4]).astype(np.uint8)
    tc = np.isin(seg, [1, 4]).astype(np.uint8)
    et = (seg == 4).astype(np.uint8)
    return np.stack([wt, tc, et], axis=0)


# ===========================================================================
# Scalar metric functions
# ===========================================================================

def compute_dice(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> float:
    """Binary Dice / F1 coefficient.

    Parameters
    ----------
    pred   : np.ndarray  Binary prediction array (any shape, flattened internally).
    target : np.ndarray  Binary ground-truth (same shape as pred).
    smooth : float       Laplace smoothing term.

    Returns
    -------
    Dice coefficient in [0, 1].  Returns 1.0 if both masks are empty.
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()

    # Edge case: both empty → perfect agreement
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def compute_iou(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> float:
    """Intersection-over-Union (Jaccard index).

    Parameters
    ----------
    pred   : np.ndarray  Binary prediction.
    target : np.ndarray  Binary ground-truth.
    smooth : float       Smoothing term.

    Returns
    -------
    IoU in [0, 1].  Returns 1.0 if both masks are empty.
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + smooth) / (union + smooth))


def compute_precision(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> float:
    """Precision (Positive Predictive Value).

    Returns
    -------
    Precision in [0, 1].  Returns 1.0 if pred is all-zero (no false positives).
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def compute_recall(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> float:
    """Recall / Sensitivity (True Positive Rate).

    Returns
    -------
    Recall in [0, 1].  Returns 1.0 if target is all-zero (nothing to detect).
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()

    if target.sum() == 0:
        return 1.0

    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def compute_specificity(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> float:
    """Specificity (True Negative Rate).

    Returns
    -------
    Specificity in [0, 1].  Returns 1.0 if background is perfectly captured.
    """
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()

    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()

    if tn + fp == 0:
        return 1.0

    return float((tn + smooth) / (tn + fp + smooth))


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance.

    Uses ``scipy.ndimage.distance_transform_edt`` for accurate surface
    distance computation.

    Parameters
    ----------
    pred   : np.ndarray  Binary 3-D prediction (D, H, W).
    target : np.ndarray  Binary 3-D ground-truth (D, H, W).

    Returns
    -------
    HD95 in voxel units, or -1.0 if scipy is unavailable or one mask is empty.
    """
    if not _SCIPY_AVAILABLE:
        warnings.warn("scipy not installed – HD95 returning -1.0", RuntimeWarning)
        return -1.0

    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if pred_bool.sum() == 0 or target_bool.sum() == 0:
        # Cannot compute surface distance for empty masks
        logger.debug("compute_hd95: one mask is empty, returning -1.0")
        return -1.0

    try:
        from scipy.ndimage import distance_transform_edt

        # Surface voxels = foreground voxels on the boundary
        def _surface_distances(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
            """Directed surface distances from A to B."""
            dist_b = distance_transform_edt(~seg_b)
            # Surface of A
            surface_a = seg_a & ~_erode(seg_a)
            return dist_b[surface_a]

        def _erode(mask: np.ndarray) -> np.ndarray:
            from scipy.ndimage import binary_erosion
            return binary_erosion(mask, iterations=1)

        d_pred_to_gt = _surface_distances(pred_bool, target_bool)
        d_gt_to_pred = _surface_distances(target_bool, pred_bool)

        if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
            return -1.0

        all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        return float(np.percentile(all_distances, 95))

    except Exception as exc:  # noqa: BLE001
        logger.warning("HD95 computation failed: %s", exc)
        return -1.0


# ===========================================================================
# SegmentationMetrics
# ===========================================================================

class SegmentationMetrics:
    """Per-batch metric computation for BraTS-style multilabel segmentation.

    Parameters
    ----------
    threshold : float
        Decision threshold applied to sigmoid probabilities.
    regions : list of str
        Tumour regions to evaluate.  Defaults to ['WT', 'TC', 'ET'].
    """

    def __init__(
        self,
        threshold: float = 0.5,
        regions: List[str] = None,
    ):
        self.threshold = threshold
        self.regions = regions if regions is not None else ["WT", "TC", "ET"]

    def compute(
        self, pred_logits: torch.Tensor, target: torch.Tensor
    ) -> dict:
        """Compute all metrics for a batch.

        Parameters
        ----------
        pred_logits : torch.Tensor  shape (B, 3, D, H, W)  – raw network logits.
        target      : torch.Tensor  shape (B, 1, D, H, W)  – integer labels 0-4.

        Returns
        -------
        dict of shape:
        {
          'WT': {'dice', 'iou', 'precision', 'recall', 'specificity'},
          'TC': { ... },
          'ET': { ... },
          'mean_dice': float,
        }
        Values are averaged over the batch dimension.
        """
        # Sigmoid → binary
        if isinstance(pred_logits, dict):
            pred_logits = pred_logits.get("logits", pred_logits)
        probs = torch.sigmoid(pred_logits.detach().cpu())
        binary_pred = (probs > self.threshold).numpy().astype(np.uint8)  # (B,3,D,H,W)

        # Convert integer labels to 3-channel binary  (B,3,D,H,W)
        target_np = target.detach().cpu().numpy()  # (B,1,D,H,W)

        batch_results: Dict[str, Dict[str, List[float]]] = {r: {
            "dice": [], "iou": [], "precision": [], "recall": [], "specificity": []
        } for r in self.regions}

        B = binary_pred.shape[0]
        for b in range(B):
            seg = target_np[b, 0]  # (D,H,W) integer mask
            gt_channels = _labels_to_regions(seg)  # (3,D,H,W)

            for ch_idx, region in enumerate(self.regions):
                pred_ch = binary_pred[b, ch_idx]
                gt_ch = gt_channels[ch_idx]

                batch_results[region]["dice"].append(compute_dice(pred_ch, gt_ch))
                batch_results[region]["iou"].append(compute_iou(pred_ch, gt_ch))
                batch_results[region]["precision"].append(compute_precision(pred_ch, gt_ch))
                batch_results[region]["recall"].append(compute_recall(pred_ch, gt_ch))
                batch_results[region]["specificity"].append(compute_specificity(pred_ch, gt_ch))

        # Average over batch
        out: dict = {}
        all_dices = []
        for region in self.regions:
            out[region] = {
                k: float(np.mean(v)) for k, v in batch_results[region].items()
            }
            all_dices.append(out[region]["dice"])

        out["mean_dice"] = float(np.mean(all_dices))
        return out

    def reset(self) -> None:
        """No-op – this class is stateless; use MetricsAccumulator for accumulation."""
        pass

    @staticmethod
    def aggregate(results_list: List[dict]) -> dict:
        """Average a list of per-sample/per-batch metric dicts.

        Parameters
        ----------
        results_list : list of dicts returned by :meth:`compute`.

        Returns
        -------
        Single dict with the same structure, values averaged over all items.
        """
        if not results_list:
            return {}

        # Collect all keys
        regions = [k for k in results_list[0].keys() if k != "mean_dice"]
        metric_keys = list(results_list[0][regions[0]].keys()) if regions else []

        aggregated: dict = {}
        for region in regions:
            aggregated[region] = {}
            for mk in metric_keys:
                vals = [r[region][mk] for r in results_list if region in r and mk in r[region]]
                aggregated[region][mk] = float(np.mean(vals)) if vals else 0.0

        # Mean dice across regions
        mean_dices = [r.get("mean_dice", 0.0) for r in results_list]
        aggregated["mean_dice"] = float(np.mean(mean_dices)) if mean_dices else 0.0

        return aggregated


# ===========================================================================
# MetricsAccumulator
# ===========================================================================

class MetricsAccumulator:
    """Running accumulator for collecting metrics across multiple batches.

    Usage
    -----
    acc = MetricsAccumulator()
    for batch in loader:
        metrics = seg_metrics.compute(preds, labels)
        acc.update(metrics)
    summary = acc.compute_mean()
    """

    def __init__(self):
        self._data: Dict[str, List[float]] = {}

    def update(self, metrics_dict: dict) -> None:
        """Add a metrics dict (from SegmentationMetrics.compute) to the accumulator.

        Nested dicts (e.g. {'WT': {'dice': 0.9}}) are flattened to 'WT/dice'.

        Parameters
        ----------
        metrics_dict : dict  Single-batch metric dict.
        """
        flat = _flatten_dict(metrics_dict)
        for key, val in flat.items():
            if key not in self._data:
                self._data[key] = []
            if isinstance(val, (int, float)):
                self._data[key].append(float(val))

    def compute_mean(self) -> dict:
        """Compute the mean over all accumulated values.

        Returns
        -------
        Flattened dict of mean metric values.
        """
        return {k: float(np.mean(v)) for k, v in self._data.items() if v}

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._data = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Recursively flatten a nested dict.

    Example
    -------
    {'WT': {'dice': 0.9}, 'mean_dice': 0.85}
    → {'WT/dice': 0.9, 'mean_dice': 0.85}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ===========================================================================
# Quick self-test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic inputs
    B, D, H, W = 2, 32, 32, 32
    pred_logits = torch.randn(B, 3, D, H, W)
    labels = torch.randint(0, 3, (B, 1, D, H, W))  # raw labels 0-2 for simplicity

    sm = SegmentationMetrics(threshold=0.5)
    result = sm.compute(pred_logits, labels)
    print("Per-region metrics:", json.dumps(result, indent=2) if False else result)
    print("mean_dice:", result["mean_dice"])

    # Accumulator test
    acc = MetricsAccumulator()
    for _ in range(5):
        acc.update(sm.compute(torch.randn(B, 3, D, H, W), labels))
    print("Accumulated mean:", acc.compute_mean())

    # Scalar function tests
    p = np.array([1, 1, 0, 0, 1])
    t = np.array([1, 0, 0, 0, 1])
    print(f"dice={compute_dice(p, t):.4f}  iou={compute_iou(p, t):.4f}  "
          f"prec={compute_precision(p, t):.4f}  rec={compute_recall(p, t):.4f}  "
          f"spec={compute_specificity(p, t):.4f}")
