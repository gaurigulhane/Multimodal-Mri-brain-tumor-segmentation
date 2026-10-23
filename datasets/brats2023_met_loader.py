"""
brats2023_met_loader.py
=======================
PyTorch Dataset and DataModule for the **BraTS 2023-MET** (Brain Metastasis)
segmentation challenge dataset.

Expected directory structure
-----------------------------
::

    <data_root>/
    ├── ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/    (optional wrapper)
    │   ├── BraTS-MET-00001-000/
    │   │   ├── BraTS-MET-00001-000_t1n.nii.gz
    │   │   ├── BraTS-MET-00001-000_t1c.nii.gz
    │   │   ├── BraTS-MET-00001-000_t2w.nii.gz
    │   │   ├── BraTS-MET-00001-000_t2f.nii.gz
    │   │   └── BraTS-MET-00001-000_seg.nii.gz
    │   └── ...
    └── ...

The loader also handles flat layouts where patient folders live directly under
*data_root*. Patient directories are discovered by looking for folders that
match the glob ``BraTS-MET-*`` or ``ASNR-MICCAI-BraTS2023-MET-*``.

Modality naming (MET vs 2021):
  MET uses t1n / t1c / t2w / t2f  instead of  t1 / t1ce / t2 / flair.
  Internally both are mapped to the same 4-channel tensor order:
  [T1, T1c, T2, FLAIR(=T2f)].
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

from .label_mapping import BraTS2023METLabelMapper, map_labels_to_regions
from .preprocessing import MultimodalPreprocessor

logger = logging.getLogger(__name__)

# Modality suffixes for BraTS 2023-MET
_SUFFIX_MAP_MET: Dict[str, str] = {
    "t1n": "_t1n.nii.gz",   # native T1
    "t1c": "_t1c.nii.gz",   # contrast-enhanced T1
    "t2w": "_t2w.nii.gz",   # T2-weighted
    "t2f": "_t2f.nii.gz",   # FLAIR (T2-FLAIR)
    "seg": "_seg.nii.gz",
}

# Internal → MultimodalPreprocessor key mapping
# MultimodalPreprocessor expects t1 / t1ce / t2 / flair
_MET_TO_STD: Dict[str, str] = {
    "t1n": "t1",
    "t1c": "t1ce",
    "t2w": "t2",
    "t2f": "flair",
}

# Glob patterns used to discover patient directories
_PATIENT_DIR_GLOBS: Tuple[str, ...] = (
    "BraTS-MET-*",
    "ASNR-MICCAI-BraTS2023-MET-*",
)


# ---------------------------------------------------------------------------
# BraTS2023METDataset
# ---------------------------------------------------------------------------

class BraTS2023METDataset(Dataset):
    """
    PyTorch :class:`~torch.utils.data.Dataset` for BraTS 2023-MET.

    Parameters
    ----------
    data_root : str
        Root directory.  The loader searches *data_root* (and one level deep)
        for patient subdirectories matching the MET naming convention.
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    transforms : callable, optional
        Transform applied to the sample dict after raw loading.  When
        ``None`` the default :class:`MultimodalPreprocessor` is used.
    use_label_mapper : bool
        If ``True``, ``"label"`` is a binary (3, D, H, W) float32 tensor
        (WT / TC / ET).  If ``False``, raw int64 (1, D, H, W) is returned.
    seed : int
        Random seed for reproducible patient-level splitting.
    train_ratio : float
        Fraction of patients for training.
    val_ratio : float
        Fraction of patients for validation; remainder → test.
    target_shape : tuple of int
        Spatial shape forwarded to :class:`MultimodalPreprocessor`.

    Returns (per sample)
    --------------------
    dict with keys:
        ``"image"``      : torch.Tensor, shape (4, D, H, W), float32
        ``"label"``      : torch.Tensor, shape (3, D, H, W) float32 or
                           (1, D, H, W) int64
        ``"raw_label"``  : torch.Tensor, shape (D, H, W), int64
        ``"patient_id"`` : str

    Examples
    --------
    >>> ds = BraTS2023METDataset("/data/BraTS2023MET", split="train")
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

        self.data_root        = Path(data_root)
        self.split            = split
        self.use_label_mapper = use_label_mapper
        self.seed             = seed
        self.train_ratio      = train_ratio
        self.val_ratio        = val_ratio
        self.target_shape     = target_shape
        self._label_mapper    = BraTS2023METLabelMapper()

        all_patients = self._discover_patients()
        if not all_patients:
            raise RuntimeError(
                f"No valid BraTS2023-MET patients found under '{data_root}'. "
                "Check that subdirectories with all 5 NIfTI files exist."
            )
        self.patients = self._split_patients(all_patients)

        self._preprocessor = MultimodalPreprocessor(
            target_shape=target_shape,
            normalize=True,
            crop=True,
        )
        self.transforms = transforms

        logger.info(
            "BraTS2023METDataset[%s]: %d patients (total=%d).",
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

        # Map MET keys to the preprocessor's expected argument names
        sample = self._preprocessor(
            t1_path    = str(paths["t1n"]),
            t1ce_path  = str(paths["t1c"]),
            t2_path    = str(paths["t2w"]),
            flair_path = str(paths["t2f"]),
            seg_path   = str(paths["seg"]),
        )

        image     = sample["image"]              # (4, D, H, W)
        raw_label = sample["label"]              # (D, H, W) int64

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
        Search *data_root* (and one extra nesting level) for patient dirs.

        Accepts directory names matching:
        - ``BraTS-MET-XXXXX-YYY``
        - ``ASNR-MICCAI-BraTS2023-MET-XXXXX-YYY``

        Returns only directories containing all 5 required NIfTI files.
        """
        candidates: List[Path] = []

        # Look directly under data_root and one level deep
        for search_dir in [self.data_root] + list(self.data_root.iterdir()
                            if self.data_root.is_dir() else []):
            if not isinstance(search_dir, Path) or not search_dir.is_dir():
                continue
            for pattern in _PATIENT_DIR_GLOBS:
                candidates.extend(sorted(search_dir.glob(pattern)))

        # De-duplicate (a dir might be found by multiple globs)
        seen: set = set()
        unique_candidates: List[Path] = []
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        valid: List[Path] = []
        for cand in sorted(unique_candidates):
            if not cand.is_dir():
                continue
            pid   = cand.name
            paths = self._build_paths(cand, pid)
            if all(p.is_file() for p in paths.values()):
                valid.append(cand)
            else:
                missing = [k for k, p in paths.items() if not p.is_file()]
                logger.warning(
                    "Patient '%s' is incomplete – missing: %s. Skipping.",
                    pid, missing,
                )

        logger.info(
            "_discover_patients (MET): %d/%d directories are valid.",
            len(valid), len(unique_candidates),
        )
        return valid

    def _split_patients(self, all_patients: List[Path]) -> List[Path]:
        """Deterministic patient-level split (same logic as BraTS2021Dataset)."""
        sorted_patients = sorted(all_patients, key=lambda p: p.name)
        rng = random.Random(self.seed)
        shuffled = sorted_patients.copy()
        rng.shuffle(shuffled)

        n     = len(shuffled)
        n_tr  = max(1, int(n * self.train_ratio))
        n_val = max(1, int(n * self.val_ratio))

        splits = {
            "train": shuffled[:n_tr],
            "val":   shuffled[n_tr : n_tr + n_val],
            "test":  shuffled[n_tr + n_val :],
        }
        logger.debug(
            "_split_patients (MET): train=%d, val=%d, test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )
        return splits[self.split]

    @staticmethod
    def _build_paths(patient_dir: Path, patient_id: str) -> Dict[str, Path]:
        """Return mapping from modality keys to expected NIfTI file paths."""
        return {
            mod: patient_dir / f"{patient_id}{suffix}"
            for mod, suffix in _SUFFIX_MAP_MET.items()
        }


# ---------------------------------------------------------------------------
# BraTS2023METDataModule
# ---------------------------------------------------------------------------

class BraTS2023METDataModule:
    """
    Convenience wrapper providing train / val / test
    :class:`~torch.utils.data.DataLoader` instances for BraTS 2023-MET.

    Parameters
    ----------
    data_root : str
        Root directory of the BraTS 2023-MET dataset.
    batch_size : int
        Samples per batch (training loader).
    num_workers : int
        Parallel data-loading workers.
    patch_size : tuple of int
        Spatial shape for :class:`MultimodalPreprocessor`.
    seed : int
        Reproducibility seed.
    pin_memory : bool
        Whether to pin memory for GPU transfers.

    Examples
    --------
    >>> dm = BraTS2023METDataModule("/data/BraTS2023MET", batch_size=2)
    >>> for batch in dm.get_train_loader():
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

        self._train_ds: Optional[BraTS2023METDataset] = None
        self._val_ds:   Optional[BraTS2023METDataset] = None
        self._test_ds:  Optional[BraTS2023METDataset] = None

    # ------------------------------------------------------------------
    # Dataset builders (lazy)
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> BraTS2023METDataset:
        return BraTS2023METDataset(
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
            batch_size=1,
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

        Label statistics are estimated from the first 5 training samples.
        """
        train_ds = self._get_dataset("train")
        val_ds   = self._get_dataset("val")
        test_ds  = self._get_dataset("test")

        info: Dict = {
            "dataset":       "BraTS2023-MET",
            "data_root":     str(self.data_root),
            "patient_counts": {
                "train": len(train_ds),
                "val":   len(val_ds),
                "test":  len(test_ds),
                "total": len(train_ds) + len(val_ds) + len(test_ds),
            },
            "image_channels": 4,
            "channel_order":  ["T1n", "T1c", "T2w", "T2f"],
            "label_regions":  ["WT", "TC", "ET"],
            "raw_labels": {
                0: "Background",
                1: "NETC",
                2: "SNFH",
                3: "ET",
            },
            "patch_size": list(self.patch_size),
        }

        label_counts = {1: 0, 2: 0, 3: 0}
        n_stats = min(5, len(train_ds))
        for i in range(n_stats):
            raw = train_ds[i]["raw_label"]
            for lv in [1, 2, 3]:
                label_counts[lv] += int((raw == lv).sum().item())

        info["label_voxel_counts_first_5_train"] = label_counts
        return info


# ---------------------------------------------------------------------------
# Synthetic sample generator
# ---------------------------------------------------------------------------

def create_synthetic_brats2023met_sample(
    output_dir: str,
    patient_id: str = "BraTS-MET-00001-000",
    shape: Tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
) -> Dict[str, str]:
    """
    Create synthetic BraTS 2023-MET NIfTI files for testing.

    Writes five ``.nii.gz`` files (t1n, t1c, t2w, t2f, seg) to
    ``<output_dir>/<patient_id>/`` using the MET naming convention.

    Parameters
    ----------
    output_dir : str
        Parent directory for the patient folder.
    patient_id : str
        Patient identifier (used for directory and file naming).
    shape : tuple of int
        Spatial shape of synthetic volumes.
    seed : int
        NumPy random seed.

    Returns
    -------
    dict
        Mapping from modality keys (``t1n``, ``t1c``, ``t2w``, ``t2f``,
        ``seg``) to absolute file paths.

    Examples
    --------
    >>> paths = create_synthetic_brats2023met_sample("/tmp/test_met")
    >>> sorted(paths.keys())
    ['seg', 't1c', 't1n', 't2f', 't2w']
    """
    rng = np.random.default_rng(seed)
    patient_dir = Path(output_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4, dtype=np.float32)

    # Spherical brain mask
    D, H, W   = shape
    cx, cy, cz = D // 2, H // 2, W // 2
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    brain_radius = min(D, H, W) // 3
    brain_mask   = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= brain_radius ** 2

    # --- Segmentation with MET label conventions ---
    seg = np.zeros(shape, dtype=np.int32)

    # SNFH ring (label 2) – analogous to edema, 55 % of brain radius
    snfh_radius = int(brain_radius * 0.55)
    snfh_mask   = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= snfh_radius ** 2
    seg[snfh_mask] = 2

    # NETC core (label 1) – 25 % of brain radius
    netc_radius = int(brain_radius * 0.25)
    netc_mask   = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= netc_radius ** 2
    seg[netc_mask] = 1

    # ET shell (label 3) – between NETC and 40 % radius
    et_radius = int(brain_radius * 0.40)
    et_outer  = (
        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    ) <= et_radius ** 2
    et_mask   = et_outer & ~netc_mask
    seg[et_mask] = 3

    saved_paths: Dict[str, str] = {}

    # MET modality keys and their base intensities
    mod_config = {
        "t1n": 300,   # native T1
        "t1c": 400,   # contrast-enhanced T1
        "t2w": 500,   # T2-weighted
        "t2f": 350,   # FLAIR
    }

    for mod, base_intensity in mod_config.items():
        data = np.zeros(shape, dtype=np.float32)
        data[brain_mask] = (
            rng.standard_normal(int(brain_mask.sum())) * 50 + base_intensity
        )
        # Contrast enhancement visible in t1c for ET region
        if mod == "t1c":
            data[et_mask] += 180.0
        # SNFH bright on FLAIR (t2f)
        if mod == "t2f":
            data[snfh_mask] += 120.0

        img   = nib.Nifti1Image(data, affine)
        fpath = str(patient_dir / f"{patient_id}{_SUFFIX_MAP_MET[mod]}")
        nib.save(img, fpath)
        saved_paths[mod] = fpath
        logger.debug("Saved synthetic MET %s → %s", mod, fpath)

    # Save segmentation
    seg_img   = nib.Nifti1Image(seg, affine)
    seg_fpath = str(patient_dir / f"{patient_id}{_SUFFIX_MAP_MET['seg']}")
    nib.save(seg_img, seg_fpath)
    saved_paths["seg"] = seg_fpath
    logger.debug("Saved synthetic MET seg → %s", seg_fpath)

    logger.info(
        "create_synthetic_brats2023met_sample: patient '%s' saved to '%s'.",
        patient_id, patient_dir,
    )
    return saved_paths


# ---------------------------------------------------------------------------
# DataLoader worker init
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
        # Create three synthetic MET patients
        for i in range(1, 4):
            pid = f"BraTS-MET-0000{i}-000"
            create_synthetic_brats2023met_sample(
                output_dir=tmpdir, patient_id=pid, shape=(32, 32, 32)
            )

        ds = BraTS2023METDataset(
            data_root=tmpdir,
            split="train",
            target_shape=(32, 32, 32),
            seed=42,
        )
        print(f"MET Train dataset length: {len(ds)}")

        sample = ds[0]
        print(f"  image  : {sample['image'].shape}, dtype={sample['image'].dtype}")
        print(f"  label  : {sample['label'].shape}, dtype={sample['label'].dtype}")
        print(f"  raw    : {sample['raw_label'].shape}")
        print(f"  pid    : {sample['patient_id']}")

        dm = BraTS2023METDataModule(
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

    print("brats2023_met_loader.py self-test PASSED.")
