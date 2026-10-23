"""
augmentation.py
===============
MONAI-based data augmentation utilities for BraTS multi-modal MRI brain
tumour segmentation.

All spatial transforms use the *same* random state for both the image and the
label dictionary key so that they remain perfectly aligned after augmentation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union

import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)

logger = logging.getLogger(__name__)

# Default keys used in the sample dict
_IMAGE_KEY = "image"
_LABEL_KEY = "label"
_ALL_KEYS  = [_IMAGE_KEY, _LABEL_KEY]


# ---------------------------------------------------------------------------
# Train augmentation
# ---------------------------------------------------------------------------

def get_train_augmentation(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    image_key: str = _IMAGE_KEY,
    label_key: str = _LABEL_KEY,
    flip_prob: float = 0.5,
    rotate90_prob: float = 0.5,
    noise_prob: float = 0.15,
    noise_std: float = 0.1,
    smooth_prob: float = 0.1,
    smooth_sigma_range: Tuple[float, float] = (0.5, 1.0),
    scale_intensity_prob: float = 0.25,
    scale_intensity_factors: Tuple[float, float] = (-0.1, 0.1),
    shift_intensity_prob: float = 0.25,
    shift_intensity_offsets: Tuple[float, float] = (-0.1, 0.1),
    zoom_prob: float = 0.2,
    zoom_range: Tuple[float, float] = (0.9, 1.1),
) -> Compose:
    """
    Build a MONAI :class:`~monai.transforms.Compose` pipeline for **training**.

    Spatial transforms (flip, rotate, zoom) are applied jointly to both
    *image* and *label* so spatial alignment is preserved.  Intensity
    transforms (noise, smooth, scale, shift) are applied to the *image* only.

    Parameters
    ----------
    patch_size : tuple of int
        Spatial dimensions of the patch – used to clamp zoom output.
    image_key : str
        Dictionary key for the image tensor (shape ``(4, D, H, W)``).
    label_key : str
        Dictionary key for the label tensor (shape ``(1, D, H, W)`` or
        ``(3, D, H, W)`` if already converted to binary regions).
    flip_prob : float
        Probability of random flip along each axis (applied per-axis).
    rotate90_prob : float
        Probability of random 90° rotation.
    noise_prob : float
        Probability of adding Gaussian noise.
    noise_std : float
        Standard deviation of Gaussian noise.
    smooth_prob : float
        Probability of Gaussian smoothing.
    smooth_sigma_range : tuple of float
        Range ``(min_sigma, max_sigma)`` for Gaussian kernel.
    scale_intensity_prob : float
        Probability of intensity scaling.
    scale_intensity_factors : tuple of float
        Additive factor range ``(min, max)`` relative to mean intensity.
    shift_intensity_prob : float
        Probability of intensity shifting.
    shift_intensity_offsets : tuple of float
        Absolute intensity shift range ``(min, max)``.
    zoom_prob : float
        Probability of random zoom.
    zoom_range : tuple of float
        Zoom factor range ``(min, max)``; values > 1.0 zoom in, < 1.0 out.

    Returns
    -------
    monai.transforms.Compose
        Composed augmentation pipeline ready to call on a sample dict.

    Examples
    --------
    >>> aug = get_train_augmentation(patch_size=(128, 128, 128))
    >>> sample = {"image": torch.zeros(4, 128, 128, 128),
    ...           "label": torch.zeros(1, 128, 128, 128)}
    >>> out = aug(sample)
    >>> out["image"].shape
    torch.Size([4, 128, 128, 128])
    """
    keys           = [image_key, label_key]
    image_only_key = [image_key]

    transforms: List = [
        # ---- Spatial augmentations (consistent across image & label) ----
        RandFlipd(
            keys=keys,
            prob=flip_prob,
            spatial_axis=0,
        ),
        RandFlipd(
            keys=keys,
            prob=flip_prob,
            spatial_axis=1,
        ),
        RandFlipd(
            keys=keys,
            prob=flip_prob,
            spatial_axis=2,
        ),
        RandRotate90d(
            keys=keys,
            prob=rotate90_prob,
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandRotate90d(
            keys=keys,
            prob=rotate90_prob,
            max_k=3,
            spatial_axes=(1, 2),
        ),
        RandZoomd(
            keys=keys,
            prob=zoom_prob,
            min_zoom=zoom_range[0],
            max_zoom=zoom_range[1],
            mode=["trilinear", "nearest"],  # image: trilinear, label: nearest
            align_corners=[True, None],
            padding_mode="constant",
        ),
        # ---- Intensity augmentations (image only) ----
        RandGaussianNoised(
            keys=image_only_key,
            prob=noise_prob,
            mean=0.0,
            std=noise_std,
        ),
        RandGaussianSmoothd(
            keys=image_only_key,
            prob=smooth_prob,
            sigma_x=smooth_sigma_range,
            sigma_y=smooth_sigma_range,
            sigma_z=smooth_sigma_range,
        ),
        RandScaleIntensityd(
            keys=image_only_key,
            factors=scale_intensity_factors,
            prob=scale_intensity_prob,
        ),
        RandShiftIntensityd(
            keys=image_only_key,
            offsets=shift_intensity_offsets,
            prob=shift_intensity_prob,
        ),
        # ---- Type enforcement ----
        EnsureTyped(keys=keys, dtype=torch.float32),
    ]

    logger.debug(
        "get_train_augmentation: built pipeline with %d transforms.", len(transforms)
    )
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Validation augmentation
# ---------------------------------------------------------------------------

def get_val_augmentation(
    image_key: str = _IMAGE_KEY,
    label_key: str = _LABEL_KEY,
) -> Compose:
    """
    Build a minimal MONAI :class:`~monai.transforms.Compose` for **validation**.

    Only ensures tensors have the correct dtype – no random operations.

    Parameters
    ----------
    image_key : str
        Dictionary key for the image tensor.
    label_key : str
        Dictionary key for the label tensor.

    Returns
    -------
    monai.transforms.Compose

    Examples
    --------
    >>> aug = get_val_augmentation()
    >>> sample = {"image": torch.zeros(4, 128, 128, 128),
    ...           "label": torch.zeros(1, 128, 128, 128)}
    >>> out = aug(sample)
    >>> out["image"].dtype
    torch.float32
    """
    keys = [image_key, label_key]
    transforms = [
        EnsureTyped(keys=keys, dtype=torch.float32),
    ]
    logger.debug("get_val_augmentation: minimal pipeline with EnsureTyped only.")
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Batch-level helper
# ---------------------------------------------------------------------------

def apply_augmentation_to_batch(
    batch_dict: Dict[str, torch.Tensor],
    augment: Compose,
    image_key: str = _IMAGE_KEY,
    label_key: str = _LABEL_KEY,
) -> Dict[str, torch.Tensor]:
    """
    Apply a MONAI augmentation pipeline to every sample in a batched dict.

    MONAI transforms operate on single-sample dicts, so this helper iterates
    over the batch dimension (axis 0) and reassembles the results.

    Parameters
    ----------
    batch_dict : dict
        Dictionary whose values are batched tensors with shape
        ``(B, C, D, H, W)`` or ``(B, D, H, W)``.
    augment : monai.transforms.Compose
        Any MONAI Compose pipeline that accepts a dict.
    image_key : str
        Key for the image tensor in *batch_dict*.
    label_key : str
        Key for the label tensor in *batch_dict*.

    Returns
    -------
    dict
        Same dictionary structure with all tensors augmented and
        stacked back into batches.  Extra keys (e.g., ``"patient_id"``)
        are carried through unchanged.

    Examples
    --------
    >>> aug = get_train_augmentation()
    >>> batch = {
    ...     "image": torch.zeros(2, 4, 128, 128, 128),
    ...     "label": torch.zeros(2, 1, 128, 128, 128),
    ... }
    >>> out = apply_augmentation_to_batch(batch, aug)
    >>> out["image"].shape
    torch.Size([2, 4, 128, 128, 128])
    """
    batch_size   = batch_dict[image_key].shape[0]
    augmented    = {k: [] for k in batch_dict}

    for i in range(batch_size):
        # Build single-sample dict
        sample: Dict[str, Union[torch.Tensor, object]] = {}
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v[i]           # shape: (C, D, H, W) or scalar
            else:
                # Non-tensor metadata (strings, etc.) – pass through
                sample[k] = v[i] if hasattr(v, "__getitem__") else v

        # Apply MONAI pipeline to image and label only
        aug_sample = augment(
            {image_key: sample[image_key], label_key: sample[label_key]}
        )

        # Reassemble
        for k in batch_dict:
            if k in (image_key, label_key):
                augmented[k].append(aug_sample[k])
            else:
                raw = batch_dict[k]
                augmented[k].append(raw[i] if isinstance(raw, torch.Tensor) else raw)

    # Stack tensors back into batches
    result: Dict[str, Union[torch.Tensor, list]] = {}
    for k, v_list in augmented.items():
        if isinstance(v_list[0], torch.Tensor):
            result[k] = torch.stack(v_list, dim=0)
        else:
            result[k] = v_list   # keep as list for non-tensor data

    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    B, C, D, H, W = 2, 4, 64, 64, 64

    train_aug = get_train_augmentation(patch_size=(D, H, W))
    val_aug   = get_val_augmentation()

    sample = {
        "image": torch.rand(C, D, H, W),
        "label": torch.randint(0, 2, (1, D, H, W)).float(),
    }
    out_t = train_aug(sample)
    print(f"Train aug image: {out_t['image'].shape}, dtype={out_t['image'].dtype}")
    print(f"Train aug label: {out_t['label'].shape}, dtype={out_t['label'].dtype}")

    out_v = val_aug(sample)
    print(f"Val   aug image: {out_v['image'].shape}, dtype={out_v['image'].dtype}")

    # Batch test
    batch = {
        "image": torch.rand(B, C, D, H, W),
        "label": torch.randint(0, 2, (B, 1, D, H, W)).float(),
    }
    out_b = apply_augmentation_to_batch(batch, train_aug)
    print(f"Batch aug image: {out_b['image'].shape}")
    print("augmentation.py self-test PASSED.")
