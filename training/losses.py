"""
losses.py
---------
Loss functions for multimodal MRI brain tumor segmentation.

Supports:
  - DiceLoss          : Soft Dice loss (sigmoid output, multilabel)
  - FocalLoss         : Focal loss for class imbalance
  - DiceFocalLoss     : Weighted combination of Dice + Focal
  - DiceCELoss        : Weighted combination of Dice + CrossEntropy (softmax)
  - TverskyLoss       : Generalised Dice with asymmetric FP/FN weighting
  - get_loss_function : Factory helper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import MONAI losses as optional back-ends (not used by default, but
# kept here so callers can fall back if desired).
# ---------------------------------------------------------------------------
try:
    from monai.losses import DiceLoss as MonaiDiceLoss  # noqa: F401
    from monai.losses import FocalLoss as MonaiFocalLoss  # noqa: F401
    from monai.losses import DiceFocalLoss as MonaiDiceFocalLoss  # noqa: F401
    from monai.losses import TverskyLoss as MonaiTverskyLoss  # noqa: F401
    _MONAI_AVAILABLE = True
    logger.info("MONAI losses available.")
except ImportError:
    _MONAI_AVAILABLE = False
    logger.info("MONAI not installed – using custom loss implementations.")


# ===========================================================================
# 1. DiceLoss
# ===========================================================================
class DiceLoss(nn.Module):
    """Soft Dice loss for multilabel segmentation with sigmoid output.

    Parameters
    ----------
    smooth : float
        Laplace smoothing term added to numerator and denominator.
    reduction : str
        'mean'  – average over batch and channels.
        'sum'   – sum over batch and channels.
        'none'  – return per-channel tensor.
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss."""
        if isinstance(pred, dict):
            pred = pred.get("logits", pred)
        assert pred.shape == target.shape, (
            f"pred and target shapes must match, got {pred.shape} vs {target.shape}"
        )

        # Flatten spatial dimensions: (B, C, N)
        B, C = pred.shape[:2]
        pred_flat = pred.contiguous().view(B, C, -1)
        target_flat = target.contiguous().view(B, C, -1).float()

        intersection = (pred_flat * target_flat).sum(dim=-1)          # (B, C)
        pred_sum = pred_flat.sum(dim=-1)                               # (B, C)
        target_sum = target_flat.sum(dim=-1)                           # (B, C)

        dice_score = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )                                                              # (B, C)
        dice_loss = 1.0 - dice_score                                   # (B, C)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss


# ===========================================================================
# 2. FocalLoss
# ===========================================================================
class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in dense segmentation.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    alpha : float
        Balancing factor for positive class (default 0.25).
    gamma : float
        Focusing exponent (default 2.0).  gamma=0 reduces to BCE.
    reduction : str
        'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        if isinstance(pred, dict):
            pred = pred.get("logits", pred)
        target = target.float()

        # Binary cross-entropy (per-element, no reduction)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Probability of the true class
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)

        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ===========================================================================
# 3. DiceFocalLoss
# ===========================================================================
class DiceFocalLoss(nn.Module):
    """Weighted combination of Dice loss and Focal loss.

    Parameters
    ----------
    dice_weight  : float  Weight for the Dice component.
    focal_weight : float  Weight for the Focal component.
    smooth       : float  Smoothing term for Dice.
    alpha        : float  Focal loss alpha.
    gamma        : float  Focal loss gamma.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1e-5,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth, reduction="mean")
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + Focal loss."""
        if isinstance(pred, dict):
            pred = pred.get("logits", pred)
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


# ===========================================================================
# 4. DiceCELoss
# ===========================================================================
class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss for softmax / multilabel formulations.

    For multilabel segmentation the CE term is computed as binary
    cross-entropy per channel (same as the focal loss with gamma=0).

    Parameters
    ----------
    dice_weight : float  Weight for Dice component.
    ce_weight   : float  Weight for CE component.
    smooth      : float  Smoothing for Dice.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + BCE loss."""
        if isinstance(pred, dict):
            pred = pred.get("logits", pred)
        target = target.float()
        pred_sigmoid = torch.sigmoid(pred)

        dice = self.dice_loss(pred_sigmoid, target)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")

        return self.dice_weight * dice + self.ce_weight * ce


# ===========================================================================
# 5. TverskyLoss
# ===========================================================================
class TverskyLoss(nn.Module):
    """Tversky loss – a generalisation of Dice that allows asymmetric weighting
    of false positives and false negatives.

    Reference: Salehi et al., "Tversky Loss Function for Image Segmentation
    Using 3D Fully Convolutional Deep Networks", MICCAI-W 2017.

    Parameters
    ----------
    alpha  : float  Weight on false positives.  (alpha=beta=0.5 → Dice)
    beta   : float  Weight on false negatives.
    smooth : float  Smoothing constant.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss.

        Parameters
        ----------
        pred   : Tensor (B, C, *) – sigmoid-activated predictions in [0, 1].
        target : Tensor (B, C, *) – binary ground-truth.

        Returns
        -------
        Scalar Tversky loss.
        """
        assert pred.shape == target.shape, (
            f"pred and target shapes must match, got {pred.shape} vs {target.shape}"
        )

        B, C = pred.shape[:2]
        pred_flat = pred.contiguous().view(B, C, -1)
        target_flat = target.contiguous().view(B, C, -1).float()

        # True positives, false positives, false negatives
        TP = (pred_flat * target_flat).sum(dim=-1)
        FP = (pred_flat * (1 - target_flat)).sum(dim=-1)
        FN = ((1 - pred_flat) * target_flat).sum(dim=-1)

        tversky_score = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return (1.0 - tversky_score).mean()


# ===========================================================================
# 6. Factory function
# ===========================================================================
def get_loss_function(loss_type: str = "dice_focal") -> nn.Module:
    """Return the requested loss module.

    Parameters
    ----------
    loss_type : str
        One of:
        - 'dice'        → DiceLoss
        - 'focal'       → FocalLoss
        - 'dice_focal'  → DiceFocalLoss  (default)
        - 'dice_ce'     → DiceCELoss
        - 'tversky'     → TverskyLoss

    Returns
    -------
    nn.Module instance (not yet moved to any device).

    Raises
    ------
    ValueError if loss_type is not recognised.
    """
    _registry = {
        "dice": DiceLoss,
        "focal": FocalLoss,
        "dice_focal": DiceFocalLoss,
        "dice_ce": DiceCELoss,
        "tversky": TverskyLoss,
    }

    loss_type = loss_type.lower().strip()
    if loss_type not in _registry:
        raise ValueError(
            f"Unknown loss_type='{loss_type}'. "
            f"Valid options: {sorted(_registry.keys())}"
        )

    loss_fn = _registry[loss_type]()
    logger.info("Using loss function: %s", type(loss_fn).__name__)
    return loss_fn


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    B, C, D, H, W = 2, 3, 16, 16, 16
    logits = torch.randn(B, C, D, H, W)
    probs = torch.sigmoid(logits)
    target = (torch.rand(B, C, D, H, W) > 0.8).float()

    for name in ("dice", "focal", "dice_focal", "dice_ce", "tversky"):
        fn = get_loss_function(name)
        if name == "dice" or name == "tversky":
            loss = fn(probs, target)
        else:
            loss = fn(logits, target)
        print(f"{name:12s}  loss = {loss.item():.6f}")
