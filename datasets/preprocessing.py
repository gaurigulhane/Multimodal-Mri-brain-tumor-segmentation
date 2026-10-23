"""
preprocessing.py
================
Complete preprocessing pipeline for multi-modal MRI brain tumour segmentation.

Key components
--------------
* ``load_nifti``               – nibabel-based NIfTI loader
* ``verify_modality_alignment``– shape consistency check across modalities
* ``normalize_intensity``      – z-score / min-max / percentile normalisation
* ``crop_foreground``          – tightest bounding-box crop around the brain
* ``MultimodalPreprocessor``   – single callable that does it all
* ``get_monai_train_transforms``/ ``get_monai_val_transforms``  – MONAI pipelines
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    ConcatItemsd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)

logger = logging.getLogger(__name__)

# Canonical modality order for the 4-channel image tensor
MODALITY_ORDER: List[str] = ["t1", "t1ce", "t2", "flair"]


# ---------------------------------------------------------------------------
# load_nifti
# ---------------------------------------------------------------------------

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file and return its data array and affine matrix.

    Parameters
    ----------
    path : str
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    data : np.ndarray
        Raw voxel data array; shape is typically ``(X, Y, Z)`` or
        ``(X, Y, Z, T)`` for 4-D volumes.
    affine : np.ndarray, shape (4, 4)
        Voxel-to-world affine transformation matrix.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    RuntimeError
        If nibabel cannot load the file.

    Examples
    --------
    >>> data, affine = load_nifti("/path/to/volume.nii.gz")
    >>> data.shape
    (240, 240, 155)
    >>> affine.shape
    (4, 4)
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    try:
        img = nib.load(path)
        data   = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine
        logger.debug("Loaded NIfTI '%s' – shape=%s, dtype=%s", path, data.shape, data.dtype)
        return data, affine
    except Exception as exc:
        raise RuntimeError(f"Failed to load NIfTI file '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# verify_modality_alignment
# ---------------------------------------------------------------------------

def verify_modality_alignment(modalities: Dict[str, np.ndarray]) -> bool:
    """
    Verify that all modality arrays share the same spatial shape.

    Parameters
    ----------
    modalities : dict
        Mapping from modality name to 3-D (or N-D) numpy array.

    Returns
    -------
    bool
        ``True`` if all arrays have identical shape.

    Raises
    ------
    ValueError
        If *modalities* is empty.

    Examples
    --------
    >>> import numpy as np
    >>> vols = {"t1": np.zeros((240, 240, 155)),
    ...         "flair": np.zeros((240, 240, 155))}
    >>> verify_modality_alignment(vols)
    True
    """
    if not modalities:
        raise ValueError("modalities dict is empty – nothing to verify.")

    shapes = {name: arr.shape for name, arr in modalities.items()}
    unique_shapes = set(shapes.values())

    if len(unique_shapes) == 1:
        logger.debug("Modality alignment OK: shape=%s", unique_shapes.pop())
        return True

    # Mismatch – log details
    logger.warning("Modality shape mismatch: %s", shapes)
    return False


# ---------------------------------------------------------------------------
# normalize_intensity
# ---------------------------------------------------------------------------

def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore",
    brain_mask: Optional[np.ndarray] = None,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
) -> np.ndarray:
    """
    Normalise the intensity of a single 3-D volume.

    Only voxels inside *brain_mask* (non-zero voxels when *brain_mask* is
    ``None``) participate in computing normalisation statistics.  This avoids
    background voxels biasing the mean / std.

    Parameters
    ----------
    volume : np.ndarray, shape (D, H, W)
        Input volume with arbitrary intensity values.
    method : str
        ``"zscore"``   – zero-mean, unit-variance normalisation.
        ``"minmax"``   – linear rescale to [0, 1].
        ``"percentile"``– clip to [*percentile_lo*, *percentile_hi*] then
                          rescale to [0, 1].
    brain_mask : np.ndarray of bool, optional
        Boolean mask of brain voxels.  When ``None``, non-zero voxels are
        treated as the brain mask.
    percentile_lo : float
        Lower percentile for ``"percentile"`` method (default 0.5 %).
    percentile_hi : float
        Upper percentile for ``"percentile"`` method (default 99.5 %).

    Returns
    -------
    np.ndarray, shape (D, H, W), dtype float32
        Normalised volume.

    Raises
    ------
    ValueError
        If *method* is not recognised.

    Examples
    --------
    >>> import numpy as np
    >>> vol = np.random.randn(64, 64, 64).astype(np.float32) * 500 + 300
    >>> out = normalize_intensity(vol, method='zscore')
    >>> abs(out[out != 0].mean()) < 0.05
    True
    """
    volume = volume.astype(np.float32)

    if brain_mask is None:
        brain_mask = volume != 0

    brain_voxels = volume[brain_mask]

    if brain_voxels.size == 0:
        logger.warning("Brain mask is empty – returning volume unchanged.")
        return volume

    if method == "zscore":
        mean = brain_voxels.mean()
        std  = brain_voxels.std()
        if std < 1e-8:
            logger.warning("Std ≈ 0 during z-score normalisation; dividing by 1.")
            std = 1.0
        out = np.zeros_like(volume, dtype=np.float32)
        out[brain_mask] = (brain_voxels - mean) / std
        return out

    elif method == "minmax":
        vmin = brain_voxels.min()
        vmax = brain_voxels.max()
        if vmax - vmin < 1e-8:
            logger.warning("Min≈Max during min-max normalisation; returning zeros.")
            return np.zeros_like(volume, dtype=np.float32)
        out = np.zeros_like(volume, dtype=np.float32)
        out[brain_mask] = (brain_voxels - vmin) / (vmax - vmin)
        return out

    elif method == "percentile":
        plo = np.percentile(brain_voxels, percentile_lo)
        phi = np.percentile(brain_voxels, percentile_hi)
        clipped = np.clip(volume, plo, phi)
        rng = phi - plo
        if rng < 1e-8:
            logger.warning("Percentile range ≈ 0; returning zeros.")
            return np.zeros_like(volume, dtype=np.float32)
        out = np.zeros_like(volume, dtype=np.float32)
        out[brain_mask] = (np.clip(brain_voxels, plo, phi) - plo) / rng
        return out

    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose from: 'zscore', 'minmax', 'percentile'."
        )


# ---------------------------------------------------------------------------
# crop_foreground
# ---------------------------------------------------------------------------

def crop_foreground(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    margin: int = 0,
) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """
    Crop *volume* to the tightest bounding box around the foreground.

    Parameters
    ----------
    volume : np.ndarray, shape (D, H, W) or (C, D, H, W)
        Input volume.  If 4-D, spatial axes are assumed to be the last three.
    mask : np.ndarray of bool, shape (D, H, W), optional
        Foreground mask.  If ``None``, non-zero voxels in *volume* define the
        foreground (for multi-channel volumes, the union across channels is
        used).
    margin : int
        Number of voxels to add around the bounding box on each side
        (clamped to valid array bounds).

    Returns
    -------
    cropped : np.ndarray
        Cropped volume; same number of dimensions as *volume*.
    crop_coords : tuple of slice
        Three :class:`slice` objects ``(d_slice, h_slice, w_slice)`` that can
        be applied to any co-registered volume to obtain the same crop.

    Examples
    --------
    >>> import numpy as np
    >>> vol = np.zeros((64, 64, 64))
    >>> vol[10:50, 10:50, 10:50] = 1.0
    >>> cropped, coords = crop_foreground(vol)
    >>> cropped.shape
    (40, 40, 40)
    """
    # Determine spatial shape
    is_multichannel = volume.ndim == 4
    if is_multichannel:
        spatial = volume.shape[1:]       # (D, H, W)
    else:
        spatial = volume.shape            # (D, H, W)

    # Build foreground mask
    if mask is None:
        if is_multichannel:
            fg = np.any(volume != 0, axis=0)
        else:
            fg = volume != 0
    else:
        fg = mask.astype(bool)

    # Find bounding box
    coords_nonzero = np.argwhere(fg)
    if coords_nonzero.size == 0:
        logger.warning("crop_foreground: foreground mask is empty – returning original.")
        full_slices = tuple(slice(0, s) for s in spatial)
        return volume, full_slices  # type: ignore[return-value]

    mins = coords_nonzero.min(axis=0)
    maxs = coords_nonzero.max(axis=0) + 1  # inclusive → exclusive

    # Apply margin
    mins = np.maximum(0, mins - margin)
    maxs = np.minimum(np.array(spatial), maxs + margin)

    d_sl = slice(int(mins[0]), int(maxs[0]))
    h_sl = slice(int(mins[1]), int(maxs[1]))
    w_sl = slice(int(mins[2]), int(maxs[2]))
    crop_coords = (d_sl, h_sl, w_sl)

    if is_multichannel:
        cropped = volume[:, d_sl, h_sl, w_sl]
    else:
        cropped = volume[d_sl, h_sl, w_sl]

    logger.debug(
        "crop_foreground: %s → %s  (slices %s)",
        volume.shape, cropped.shape, crop_coords,
    )
    return cropped, crop_coords  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MultimodalPreprocessor
# ---------------------------------------------------------------------------

class MultimodalPreprocessor:
    """
    End-to-end preprocessing for a single BraTS patient.

    Calling an instance with four modality paths (and optionally a
    segmentation path) returns a dict ready to be fed into a DataLoader.

    Parameters
    ----------
    target_shape : tuple of int
        Desired spatial shape ``(D, H, W)`` after resizing.  Currently
        implemented via nibabel resampling (MONAI ``Resize`` can be
        substituted if preferred).
    normalize : bool
        Whether to apply per-modality z-score normalisation.
    crop : bool
        Whether to crop the brain from the background.
    norm_method : str
        Normalisation method forwarded to :func:`normalize_intensity`.

    Examples
    --------
    >>> prep = MultimodalPreprocessor(target_shape=(128, 128, 128))
    >>> sample = prep(t1_path, t1ce_path, t2_path, flair_path, seg_path)
    >>> sample["image"].shape
    torch.Size([4, 128, 128, 128])
    >>> sample["label"].shape
    torch.Size([128, 128, 128])
    """

    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        crop: bool = True,
        norm_method: str = "zscore",
    ) -> None:
        self.target_shape = target_shape
        self.normalize    = normalize
        self.crop         = crop
        self.norm_method  = norm_method

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        flair_path: str,
        seg_path: Optional[str] = None,
    ) -> Dict:
        """
        Full preprocessing pipeline for one patient.

        Parameters
        ----------
        t1_path, t1ce_path, t2_path, flair_path : str
            Paths to the four MRI modality NIfTI files.
        seg_path : str, optional
            Path to the segmentation NIfTI.  When provided, the label tensor
            is included in the output dict.

        Returns
        -------
        dict with keys:
            ``"image"``   : torch.Tensor, shape (4, D, H, W), float32
            ``"label"``   : torch.Tensor, shape (D, H, W), int64 (only if *seg_path*)
            ``"metadata"``  : dict with affines, shape history, patient file paths
        """
        # 1. Load & validate
        modalities, affines, seg, metadata = self._load_and_validate(
            t1_path, t1ce_path, t2_path, flair_path, seg_path
        )

        # 2. Crop foreground (use union of all modalities as mask)
        if self.crop:
            union_mask = np.any(
                np.stack(list(modalities.values()), axis=0) != 0, axis=0
            )
            ref_key = MODALITY_ORDER[0]
            _, crop_coords = crop_foreground(modalities[ref_key], mask=union_mask)
            modalities = {
                k: v[crop_coords[0], crop_coords[1], crop_coords[2]]
                for k, v in modalities.items()
            }
            if seg is not None:
                seg = seg[crop_coords[0], crop_coords[1], crop_coords[2]]
            metadata["crop_coords"] = str(crop_coords)

        # 3. Resize to target_shape (simple numpy zoom)
        modalities, seg = self._resize(modalities, seg)
        metadata["target_shape"] = self.target_shape

        # 4. Normalise (per modality on brain voxels)
        if self.normalize:
            modalities = self._normalize(modalities)

        # 5. Stack into 4-channel tensor
        image_tensor, label_tensor = self._stack_modalities(modalities, seg)

        out: Dict = {
            "image":    image_tensor,
            "metadata": metadata,
        }
        if label_tensor is not None:
            out["label"] = label_tensor

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_validate(
        self,
        t1_path: str,
        t1ce_path: str,
        t2_path: str,
        flair_path: str,
        seg_path: Optional[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray], Dict]:
        """Load NIfTIs, verify alignment, and return raw arrays + metadata."""
        paths = {
            "t1":    t1_path,
            "t1ce": t1ce_path,
            "t2":   t2_path,
            "flair": flair_path,
        }
        modalities: Dict[str, np.ndarray] = {}
        affines:    Dict[str, np.ndarray] = {}

        for name, p in paths.items():
            data, affine = load_nifti(str(p))
            modalities[name] = data
            affines[name]    = affine

        if not verify_modality_alignment(modalities):
            raise ValueError(
                "Modality arrays have different shapes – cannot proceed. "
                f"Shapes: { {k: v.shape for k, v in modalities.items()} }"
            )

        seg: Optional[np.ndarray] = None
        if seg_path is not None:
            seg_data, seg_affine = load_nifti(str(seg_path))
            seg = seg_data.astype(np.int32)
            affines["seg"] = seg_affine

        metadata = {
            "original_shape": list(modalities["t1"].shape),
            "paths":          {k: str(v) for k, v in {**paths, "seg": seg_path}.items()},
            "affines":        {k: v.tolist() for k, v in affines.items()},
        }
        return modalities, affines, seg, metadata

    def _normalize(
        self, modalities: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply per-modality z-score normalisation on brain (non-zero) voxels."""
        normalised: Dict[str, np.ndarray] = {}
        for name, vol in modalities.items():
            normalised[name] = normalize_intensity(vol, method=self.norm_method)
        return normalised

    def _resize(
        self,
        modalities: Dict[str, np.ndarray],
        seg: Optional[np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Resize all volumes to ``self.target_shape`` using scipy zoom.

        Linear interpolation for images, nearest-neighbour for segmentation.
        """
        try:
            from scipy.ndimage import zoom as scipy_zoom
        except ImportError:
            logger.warning(
                "scipy not available – skipping resize step. "
                "Install scipy or use MONAI Spacingd for resampling."
            )
            return modalities, seg

        sample_shape = list(modalities[MODALITY_ORDER[0]].shape)
        zoom_factors = [
            self.target_shape[i] / sample_shape[i] for i in range(3)
        ]

        if zoom_factors == [1.0, 1.0, 1.0]:
            return modalities, seg

        resized: Dict[str, np.ndarray] = {}
        for name, vol in modalities.items():
            resized[name] = scipy_zoom(
                vol, zoom_factors, order=1, mode="constant", cval=0
            ).astype(np.float32)

        seg_resized: Optional[np.ndarray] = None
        if seg is not None:
            seg_resized = scipy_zoom(
                seg, zoom_factors, order=0, mode="constant", cval=0
            ).astype(np.int32)

        return resized, seg_resized

    def _stack_modalities(
        self,
        modalities: Dict[str, np.ndarray],
        seg: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Stack modalities in canonical order → 4-channel float32 tensor."""
        channels = [
            torch.from_numpy(modalities[m]) for m in MODALITY_ORDER
        ]
        image_tensor = torch.stack(channels, dim=0).float()   # (4, D, H, W)

        label_tensor: Optional[torch.Tensor] = None
        if seg is not None:
            label_tensor = torch.from_numpy(seg).long()        # (D, H, W)

        return image_tensor, label_tensor


# ---------------------------------------------------------------------------
# MONAI transform pipelines
# ---------------------------------------------------------------------------

def get_monai_train_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    image_keys: List[str] = None,
    seg_key:    str = "seg",
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    flip_prob: float = 0.5,
    noise_prob: float = 0.15,
    noise_std: float = 0.1,
    rotate90_prob: float = 0.5,
) -> Compose:
    """
    Build a MONAI :class:`~monai.transforms.Compose` pipeline for **training**.

    Expected input dict keys: ``t1``, ``t1ce``, ``t2``, ``flair``, ``seg``
    (plain NIfTI paths or pre-loaded arrays from
    :func:`monai.transforms.LoadImaged`).

    The pipeline:
    1. ``EnsureChannelFirstd``
    2. ``Orientationd`` (RAS)
    3. ``Spacingd`` (1 mm isotropic)
    4. ``NormalizeIntensityd`` per-channel (non-zero mask)
    5. ``ConcatItemsd`` → merges t1/t1ce/t2/flair into ``image`` key
    6. ``RandSpatialCropd``
    7. ``RandFlipd`` (× 3 axes)
    8. ``RandRotate90d``
    9. ``RandGaussianNoised``
    10. ``ToTensord``

    Parameters
    ----------
    patch_size : tuple of int
        Spatial dimensions for random patch extraction.
    image_keys : list of str, optional
        Modality keys.  Defaults to ``["t1", "t1ce", "t2", "flair"]``.
    seg_key : str
        Dictionary key for the segmentation volume.
    spacing : tuple of float
        Target isotropic voxel spacing in mm.
    flip_prob : float
        Probability of random flip per axis.
    noise_prob : float
        Probability of Gaussian noise addition.
    noise_std : float
        Standard deviation of Gaussian noise.
    rotate90_prob : float
        Probability of random 90-degree rotation.

    Returns
    -------
    monai.transforms.Compose

    Examples
    --------
    >>> tfm = get_monai_train_transforms(patch_size=(128, 128, 128))
    >>> sample = {"t1": "path/t1.nii.gz", ...}  # loaded by LoadImaged
    """
    if image_keys is None:
        image_keys = ["t1", "t1ce", "t2", "flair"]

    all_keys    = image_keys + [seg_key]
    spatial_keys = all_keys  # flip/crop applied to everything

    transforms = [
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=spacing,
            mode=["bilinear"] * len(image_keys) + ["nearest"],
        ),
        NormalizeIntensityd(
            keys=image_keys,
            nonzero=True,   # compute stats only over non-zero (brain) voxels
            channel_wise=True,
        ),
        # Merge modality channels: [t1, t1ce, t2, flair] → image (4, D, H, W)
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        # Spatial augmentations
        RandSpatialCropd(
            keys=["image", seg_key],
            roi_size=list(patch_size),
            random_size=False,
        ),
        RandFlipd(keys=["image", seg_key], prob=flip_prob, spatial_axis=0),
        RandFlipd(keys=["image", seg_key], prob=flip_prob, spatial_axis=1),
        RandFlipd(keys=["image", seg_key], prob=flip_prob, spatial_axis=2),
        RandRotate90d(
            keys=["image", seg_key],
            prob=rotate90_prob,
            max_k=3,
            spatial_axes=(0, 1),
        ),
        # Intensity augmentations (image only)
        RandGaussianNoised(keys=["image"], prob=noise_prob, mean=0.0, std=noise_std),
        ToTensord(keys=["image", seg_key]),
    ]

    logger.debug(
        "get_monai_train_transforms: %d transforms, patch_size=%s",
        len(transforms), patch_size,
    )
    return Compose(transforms)


def get_monai_val_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    image_keys: List[str] = None,
    seg_key:    str = "seg",
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Build a MONAI :class:`~monai.transforms.Compose` pipeline for **validation**.

    Identical to the training pipeline but without random augmentations and
    without ``RandSpatialCropd`` (the full volume is processed).

    Parameters
    ----------
    patch_size : tuple of int
        Unused – kept for API symmetry with :func:`get_monai_train_transforms`.
    image_keys : list of str, optional
        Modality keys; defaults to ``["t1", "t1ce", "t2", "flair"]``.
    seg_key : str
        Dictionary key for the segmentation volume.
    spacing : tuple of float
        Target isotropic voxel spacing in mm.

    Returns
    -------
    monai.transforms.Compose

    Examples
    --------
    >>> tfm = get_monai_val_transforms()
    """
    if image_keys is None:
        image_keys = ["t1", "t1ce", "t2", "flair"]

    all_keys = image_keys + [seg_key]

    transforms = [
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=spacing,
            mode=["bilinear"] * len(image_keys) + ["nearest"],
        ),
        NormalizeIntensityd(
            keys=image_keys,
            nonzero=True,
            channel_wise=True,
        ),
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        ToTensord(keys=["image", seg_key]),
    ]

    logger.debug(
        "get_monai_val_transforms: %d transforms (no random ops).", len(transforms)
    )
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.DEBUG)

    # Create a tiny synthetic NIfTI for testing
    shape = (32, 32, 32)
    affine = np.eye(4)
    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths: Dict[str, str] = {}
        for mod in ["t1", "t1ce", "t2", "flair"]:
            data  = rng.standard_normal(shape).astype(np.float32) * 200 + 400
            img   = nib.Nifti1Image(data, affine)
            p     = os.path.join(tmpdir, f"{mod}.nii.gz")
            nib.save(img, p)
            paths[mod] = p

        seg_data = rng.integers(0, 4, size=shape).astype(np.int32)
        seg_img  = nib.Nifti1Image(seg_data, affine)
        seg_path = os.path.join(tmpdir, "seg.nii.gz")
        nib.save(seg_img, seg_path)

        # Test individual functions
        vol, aff = load_nifti(paths["t1"])
        assert vol.shape == shape, f"Unexpected shape: {vol.shape}"
        print(f"load_nifti OK: shape={vol.shape}")

        mods = {m: load_nifti(paths[m])[0] for m in ["t1", "t1ce", "t2", "flair"]}
        assert verify_modality_alignment(mods)
        print("verify_modality_alignment OK")

        normed = normalize_intensity(vol, method="zscore")
        brain  = normed[normed != 0]
        print(f"normalize_intensity OK: mean={brain.mean():.4f}, std={brain.std():.4f}")

        cropped, coords = crop_foreground(vol)
        print(f"crop_foreground OK: {vol.shape} → {cropped.shape}")

        # Test MultimodalPreprocessor
        prep = MultimodalPreprocessor(target_shape=(32, 32, 32), crop=False)
        sample = prep(
            paths["t1"], paths["t1ce"], paths["t2"], paths["flair"], seg_path
        )
        print(f"MultimodalPreprocessor OK: image={sample['image'].shape}, label={sample['label'].shape}")

    print("\npreprocessing.py self-test PASSED.")
