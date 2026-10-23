"""
brats2021_loader.py
===================
PyTorch Dataset and DataModule for the **BraTS 2021** glioma segmentation
challenge dataset.

Expected directory structure
-----------------------------
::

    <data_root>/
    ├── BraTS2021_00001/
    │   ├── BraTS2021_00001_t1.nii.gz
    │   ├── BraTS2021_00001_t1ce.nii.gz
    │   ├── BraTS2021_00001_t2.nii.gz
    │   ├── BraTS2021_00001_flair.nii.gz
    │   └── BraTS2021_00001_seg.nii.gz
    ├── BraTS2021_00002/
    │   └── ...
    └── ...

Each patient subdirectory must contain all five NIfTI files.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .label_mapping import BraTS2021LabelMapper, map_labels_to_regions
from .preprocessing import (
    MultimodalPreprocessor,
    get_monai_train_transforms,
    get_monai_val_transforms,
)

logger = logging.getLogger(__name__)

# Modality suffixes for BraTS 2021
_SUFFIX_MAP: Dict[str, str] = {
    "t1":    "_t1.nii.gz",
    "t1ce":  "_t1ce.nii.gz",
    "t2":    "_t2.nii.gz",
    "flair": "_flair.nii.gz",
    "seg":   "_seg.nii.gz",
}


# ---------------------------------------------------------------------------
# BraTS2021Dataset
# ---------------------------------------------------------------------------

class BraTS2021Dataset(Dataset):
    """
    PyTorch :class:`~torch.utils.data.Dataset` for BraTS 2021.

    Parameters
    ----------
    data_root : str
        Root directory containing per-patient subdirectories.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    transforms : callable, optional
        A transform / MONAI Compose applied to the sample dict **after** raw
        loading.  When ``None`` the default :class:`MultimodalPreprocessor`
        is used.
    use_label_mapper : bool
        If ``True``, the ``"label"`` key in the returned dict is a *binary*
        tensor of shape ``(3, D, H, W)`` (WT / TC / ET channels) produced by
        :class:`BraTS2021LabelMapper`.  If ``False`` the raw integer tensor
        ``(1, D, H, W)`` is returned.
    seed : int
        Random seed for reproducible patient-level splitting.
    train_ratio : float
        Fraction of patients assigned to the training split.
    val_ratio : float
        Fraction of patients assigned to the validation split.
        The remainder goes to the test split.

    Returns (per sample)
    --------------------
    dict with keys:
        ``"image"``      : torch.Tensor, shape (4, D, H, W), float32
        ``"label"``      : torch.Tensor, shape (3, D, H, W) float32 or
                           (1, D, H, W) int64 depending on *use_label_mapper*
        ``"raw_label"``  : torch.Tensor, shape (D, H, W), int64  (always raw)
        ``"patient_id"`` : str

    Examples
    --------
    >>> ds = BraTS2021Dataset("/data/BraTS2021", split="train")
    >>> sample = ds[0]
    >>> sample["image"].shape
    torch.Size([4, 128, 128, 128])
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        use_label_mapper: bool = True,
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'.")

        self.data_root       = Path(data_root)
        self.split           = split
        self.use_label_mapper = use_label_mapper
        self.seed            = seed
        self.train_ratio     = train_ratio
        self.val_ratio       = val_ratio
        self.target_shape    = target_shape
        self._label_mapper   = BraTS2021LabelMapper()

        # Build patient list
        all_patients = self._discover_patients()
        if not all_patients:
            raise RuntimeError(
                f"No valid BraTS2021 patients found under '{data_root}'. "
                "Check that subdirectories exist with all 5 NIfTI files."
            )
        self.patients = self._split_patients(all_patients)

        # Default preprocessor (used when transforms=None)
        self._preprocessor = MultimodalPreprocessor(
            target_shape=target_shape,
            normalize=True,
            crop=True,
        )
        self.transforms = transforms

        logger.info(
            "BraTS2021Dataset[%s]: %d patients found (total=%d).",
            split, len(self.patients), len(all_patients),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        patient_dir = self.patients[idx]
        patient_id  = patient_dir.name
        paths       = self._build_paths(patient_dir, patient_id)

        # Load raw arrays
        sample = self._preprocessor(
            t1_path    = str(paths["t1"]),
            t1ce_path  = str(paths["t1ce"]),
            t2_path    = str(paths["t2"]),
            flair_path = str(paths["flair"]),
            seg_path   = str(paths["seg"]),
        )

        image     = sample["image"]              # (4, D, H, W) float32
        raw_label = sample["label"]              # (D, H, W)   int64

        if self.use_label_mapper:
            label = self._label_mapper(raw_label)  # (3, D, H, W) float32
        else:
            label = raw_label.unsqueeze(0)         # (1, D, H, W) int64

        out: Dict[str, Union[torch.Tensor, str]] = {
            "image":      image,
            "label":      label,
            "raw_label":  raw_label,
            "patient_id": patient_id,
        }

        if self.transforms is not None:
            out = self.transforms(out)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_patients(self) -> List[Path]:
        """
        Glob for patient directories and return those that have all 5 files.

        A valid patient directory must match the pattern ``BraTS2021_*``
        and contain all of: ``*_t1.nii.gz``, ``*_t1ce.nii.gz``,
        ``*_t2.nii.gz``, ``*_flair.nii.gz``, ``*_seg.nii.gz``.
        """
        valid: List[Path] = []
        candidates = sorted(self.data_root.glob("BraTS2021_*"))

        for cand in candidates:
            if not cand.is_dir():
                continue
            pid    = cand.name
            paths  = self._build_paths(cand, pid)
            if all(p.is_file() for p in paths.values()):
                valid.append(cand)
            else:
                missing = [k for k, p in paths.items() if not p.is_file()]
                logger.warning(
                    "Patient '%s' is incomplete – missing: %s. Skipping.",
                    pid, missing,
                )

        logger.info(
            "_discover_patients: %d/%d directories are valid.",
            len(valid), len(candidates),
        )
        return valid

    def _split_patients(self, all_patients: List[Path]) -> List[Path]:
        """
        Deterministic patient-level split.

        Sorts patients by name, shuffles with *seed*, then slices.
        """
        sorted_patients = sorted(all_patients, key=lambda p: p.name)
        rng = random.Random(self.seed)
        shuffled = sorted_patients.copy()
        rng.shuffle(shuffled)

        n     = len(shuffled)
        n_tr  = max(1, int(n * self.train_ratio))
        n_val = max(1, int(n * self.val_ratio))
        # Remaining go to test (at least 1 if possible)
        n_te  = max(0, n - n_tr - n_val)

        splits = {
            "train": shuffled[:n_tr],
            "val":   shuffled[n_tr : n_tr + n_val],
            "test":  shuffled[n_tr + n_val :],
        }

        logger.debug(
            "_split_patients: train=%d, val=%d, test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )
        return splits[self.split]

    @staticmethod
    def _build_paths(patient_dir: Path, patient_id: str) -> Dict[str, Path]:
        """Return a dict mapping modality keys to expected NIfTI paths."""
        return {
            mod: patient_dir / f"{patient_id}{suffix}"
            for mod, suffix in _SUFFIX_MAP.items()
        }


# ---------------------------------------------------------------------------
# BraTS2021DataModule
# ---------------------------------------------------------------------------

class BraTS2021DataModule:
    """
    Convenience wrapper that creates train / val / test
    :class:`~torch.utils.data.DataLoader` instances for BraTS 2021.

    Parameters
    ----------
    data_root : str
        Root directory of the BraTS 2021 dataset.
    batch_size : int
        Samples per batch.
    num_workers : int
        Parallel data-loading workers.
    patch_size : tuple of int
        Spatial shape passed to MONAI transforms and
        :class:`MultimodalPreprocessor`.
    seed : int
        Reproducibility seed forwarded to :class:`BraTS2021Dataset`.
    pin_memory : bool
        Whether to use pinned (page-locked) memory for GPU transfers.

    Examples
    --------
    >>> dm = BraTS2021DataModule("/data/BraTS2021", batch_size=2)
    >>> train_loader = dm.get_train_loader()
    >>> for batch in train_loader:
    ...     print(batch["image"].shape)
    ...     break
    """

    def __init__(
        self,
        data_root:   str,
        batch_size:  int = 1,
        num_workers: int = 0,
        patch_size:  Tuple[int, int, int] = (128, 128, 128),
        seed:        int = 42,
        pin_memory:  bool = False,
    ) -> None:
        self.data_root   = data_root
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.patch_size  = patch_size
        self.seed        = seed
        self.pin_memory  = pin_memory

        self._train_ds: Optional[BraTS2021Dataset] = None
        self._val_ds:   Optional[BraTS2021Dataset] = None
        self._test_ds:  Optional[BraTS2021Dataset] = None

    # ------------------------------------------------------------------
    # Dataset builders (lazy)
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> BraTS2021Dataset:
        return BraTS2021Dataset(
            data_root=self.data_root,
            split=split,
            transforms=None,
            use_label_mapper=True,
            seed=self.seed,
            target_shape=self.patch_size,
        )

    # ------------------------------------------------------------------
    # Public DataLoader accessors
    # ------------------------------------------------------------------

    def get_train_loader(self) -> DataLoader:
        """Return the training DataLoader with shuffling enabled."""
        if self._train_ds is None:
            self._train_ds = self._get_dataset("train")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
        )

    def get_val_loader(self) -> DataLoader:
        """Return the validation DataLoader (no shuffling)."""
        if self._val_ds is None:
            self._val_ds = self._get_dataset("val")
        return DataLoader(
            self._val_ds,
            batch_size=1,  # full volumes – keep batch=1 for sliding window inference
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_test_loader(self) -> DataLoader:
        """Return the test DataLoader (no shuffling)."""
        if self._test_ds is None:
            self._test_ds = self._get_dataset("test")
        return DataLoader(
            self._test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_dataset_info(self) -> Dict:
        """
        Return a summary dict with patient counts and basic label statistics.

        Label statistics are estimated from the first 10 training samples to
        avoid loading the entire dataset.
        """
        train_ds = self._get_dataset("train")
        val_ds   = self._get_dataset("val")
        test_ds  = self._get_dataset("test")

        info: Dict = {
            "dataset":       "BraTS2021",
            "data_root":     str(self.data_root),
            "patient_counts": {
                "train": len(train_ds),
                "val":   len(val_ds),
                "test":  len(test_ds),
                "total": len(train_ds) + len(val_ds) + len(test_ds),
            },
            "image_channels": 4,
            "channel_order":  ["T1", "T1ce", "T2", "FLAIR"],
            "label_regions":  ["WT", "TC", "ET"],
            "raw_labels":     {0: "Background", 1: "NCR", 2: "ED", 3: "ET"},
            "patch_size":     list(self.patch_size),
        }

        # Quick label stats from a few training samples
        label_counts = {1: 0, 2: 0, 3: 0}
        n_stats = min(5, len(train_ds))
        for i in range(n_stats):
            raw = train_ds[i]["raw_label"]
            for lv in [1, 2, 3]:
                label_counts[lv] += int((raw == lv).sum().item())

        info["label_voxel_counts_first_5_train"] = label_counts
        return info


# ---------------------------------------------------------------------------
# Synthetic sample generator (for unit tests / CI)
# ---------------------------------------------------------------------------

def create_synthetic_brats2021_sample(
    output_dir: str,
    patient_id: str = "BraTS2021_00000",
    shape: Tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
) -> Dict[str, str]:
    """
    Create synthetic BraTS 2021 NIfTI files for testing.

    Writes five ``.nii.gz`` files (t1, t1ce, t2, flair, seg) to
    ``<output_dir>/<patient_id>/``.

    Parameters
    ----------
    output_dir : str
        Parent directory under which the patient folder is created.
    patient_id : str
        Patient identifier used for directory and file naming.
    shape : tuple of int
        Spatial shape of the synthetic volumes.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    dict
        Mapping from modality keys to absolute file paths of the saved NIfTIs.

    Examples
    --------
    >>> paths = create_synthetic_brats2021_sample("/tmp/test_data")
    >>> list(paths.keys())
    ['t1', 't1ce', 't2', 'flair', 'seg']
    """
    rng = np.random.default_rng(seed)
    patient_dir = Path(output_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4, dtype=np.float32)
    # 1 mm isotropic voxels along diagonal
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0

    # --- Create a simple spherical brain-like mask ---
    D, H, W   = shape
    cx, cy, cz = D // 2, H // 2, W // 2
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    brain_radius = min(D, H, W) // 3
    brain_mask = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= brain_radius ** 2

    # --- Segmentation: small concentric tumour regions ---
    seg = np.zeros(shape, dtype=np.int32)
    # ED ring (label 2): 60 % of brain radius
    ed_radius = int(brain_radius * 0.55)
    ed_mask   = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= ed_radius ** 2
    seg[ed_mask] = 2

    # NCR core (label 1): 30 % of brain radius
    ncr_radius = int(brain_radius * 0.30)
    ncr_mask   = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= ncr_radius ** 2
    seg[ncr_mask] = 1

    # ET ring (label 3): between NCR and a slightly larger sphere
    et_radius = int(brain_radius * 0.40)
    et_outer  = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= et_radius ** 2
    et_mask   = et_outer & ~ncr_mask
    seg[et_mask] = 3

    saved_paths: Dict[str, str] = {}

    for mod in ["t1", "t1ce", "t2", "flair"]:
        base_intensity = {"t1": 300, "t1ce": 400, "t2": 500, "flair": 350}[mod]
        data = np.zeros(shape, dtype=np.float32)
        # Brain tissue signal
        data[brain_mask] = (
            rng.standard_normal(int(brain_mask.sum())) * 50 + base_intensity
        )
        # Tumour enhancement in t1ce
        if mod == "t1ce":
            data[et_mask] += 150.0
        # FLAIR hyperintensity in edema
        if mod == "flair":
            data[ed_mask] += 100.0

        img  = nib.Nifti1Image(data, affine)
        fpath = str(patient_dir / f"{patient_id}_{mod}.nii.gz")
        nib.save(img, fpath)
        saved_paths[mod] = fpath
        logger.debug("Saved synthetic %s → %s", mod, fpath)

    # Save segmentation
    seg_img   = nib.Nifti1Image(seg, affine)
    seg_fpath = str(patient_dir / f"{patient_id}_seg.nii.gz")
    nib.save(seg_img, seg_fpath)
    saved_paths["seg"] = seg_fpath
    logger.debug("Saved synthetic seg → %s", seg_fpath)

    logger.info(
        "create_synthetic_brats2021_sample: patient '%s' saved to '%s'.",
        patient_id, patient_dir,
    )
    return saved_paths


# ---------------------------------------------------------------------------
# DataLoader worker init (for reproducibility)
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two synthetic patients
        for pid in ["BraTS2021_00000", "BraTS2021_00001", "BraTS2021_00002"]:
            create_synthetic_brats2021_sample(
                output_dir=tmpdir, patient_id=pid, shape=(32, 32, 32)
            )

        ds = BraTS2021Dataset(
            data_root=tmpdir,
            split="train",
            target_shape=(32, 32, 32),
            seed=42,
        )
        print(f"Train dataset length: {len(ds)}")

        sample = ds[0]
        print(f"  image  : {sample['image'].shape}, dtype={sample['image'].dtype}")
        print(f"  label  : {sample['label'].shape}, dtype={sample['label'].dtype}")
        print(f"  raw    : {sample['raw_label'].shape}")
        print(f"  pid    : {sample['patient_id']}")

        dm = BraTS2021DataModule(
            data_root=tmpdir,
            batch_size=1,
            num_workers=0,
            patch_size=(32, 32, 32),
        )
        info = dm.get_dataset_info()
        print(json.dumps(info, indent=2, default=str))

        loader = dm.get_train_loader()
        for batch in loader:
            print(f"  batch image: {batch['image'].shape}")
            break

    print("brats2021_loader.py self-test PASSED.")
