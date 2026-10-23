"""
label_mapping.py
================
BraTS Label Conventions:

BraTS 2021 (Glioma):
  Raw labels: 0=Background, 1=NCR (Necrotic Core), 2=ED (Edema), 3=ET (Enhancing Tumor)
  Derived regions:
    WT (Whole Tumor) = labels 1+2+3
    TC (Tumor Core) = labels 1+3
    ET (Enhancing Tumor) = label 3

BraTS 2023-MET (Metastasis):
  Raw labels: 0=Background, 1=NETC (Non-Enhancing Tumor Core),
              2=SNFH (Surrounding Non-Enhancing FLAIR Hyperintensity), 3=ET (Enhancing Tumor)
  Derived regions:
    WT (Whole Tumor) = labels 1+2+3
    TC (Tumor Core) = labels 1+3
    ET (Enhancing Tumor) = label 3
  Note: SNFH in MET ~corresponds to ED in glioma but has different clinical meaning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LabelConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class LabelConfig:
    """
    Configuration object that describes the label semantics for a BraTS dataset.

    Attributes
    ----------
    dataset_name : str
        Human-readable name, e.g. ``"BraTS2021"`` or ``"BraTS2023-MET"``.
    raw_label_values : List[int]
        Integer values that may appear in the raw segmentation NIfTI, including 0.
    region_names : List[str]
        Ordered list of output region names; the channel order in the binary mask
        follows this list, e.g. ``["WT", "TC", "ET"]``.
    region_definitions : Dict[str, List[int]]
        Maps each region name to the raw label values that belong to it, e.g.
        ``{"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}``.
    raw_label_names : Dict[int, str]
        Human-readable name for every raw integer label value.
    clinical_notes : str
        Free-form text with dataset-specific clinical context.
    """

    dataset_name: str
    raw_label_values: List[int]
    region_names: List[str]
    region_definitions: Dict[str, List[int]]
    raw_label_names: Dict[int, str] = field(default_factory=dict)
    clinical_notes: str = ""

    def num_regions(self) -> int:
        """Return the number of output binary regions / channels."""
        return len(self.region_names)

    def __repr__(self) -> str:
        lines = [
            f"LabelConfig(dataset='{self.dataset_name}')",
            f"  raw labels : {self.raw_label_values}",
        ]
        for rname in self.region_names:
            lines.append(f"  {rname:6s} <- raw {self.region_definitions[rname]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predefined configs
# ---------------------------------------------------------------------------

BRATS2021_CONFIG = LabelConfig(
    dataset_name="BraTS2021",
    raw_label_values=[0, 1, 2, 3],
    region_names=["WT", "TC", "ET"],
    region_definitions={
        "WT": [1, 2, 3],   # Whole Tumour  = NCR + ED + ET
        "TC": [1, 3],       # Tumour Core   = NCR + ET
        "ET": [3],          # Enhancing Tumour
    },
    raw_label_names={
        0: "Background",
        1: "NCR (Necrotic Core / Non-Enhancing Tumour Core)",
        2: "ED (Peritumoral Edema)",
        3: "ET (Enhancing Tumour)",
    },
    clinical_notes=(
        "Glioma segmentation challenge. "
        "Label 4 may appear in some older BraTS editions for ET; "
        "BraTS 2021 uses label 3 for ET."
    ),
)

BRATS2023MET_CONFIG = LabelConfig(
    dataset_name="BraTS2023-MET",
    raw_label_values=[0, 1, 2, 3],
    region_names=["WT", "TC", "ET"],
    region_definitions={
        "WT": [1, 2, 3],   # Whole Tumour  = NETC + SNFH + ET
        "TC": [1, 3],       # Tumour Core   = NETC + ET
        "ET": [3],          # Enhancing Tumour
    },
    raw_label_names={
        0: "Background",
        1: "NETC (Non-Enhancing Tumour Core)",
        2: "SNFH (Surrounding Non-Enhancing FLAIR Hyperintensity)",
        3: "ET (Enhancing Tumour)",
    },
    clinical_notes=(
        "Brain metastasis segmentation challenge (BraTS 2023). "
        "SNFH corresponds roughly to ED in glioma but has distinct clinical meaning "
        "in the metastasis context. TC = NETC + ET (no SNFH, unlike WT)."
    ),
)


# ---------------------------------------------------------------------------
# Base mapper
# ---------------------------------------------------------------------------

class _BaseLabelMapper:
    """
    Internal base class for label mappers.

    Subclasses must set ``config : LabelConfig`` as a class attribute.
    """

    config: LabelConfig  # must be overridden

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        label: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert a raw integer label map to a 3-channel binary mask.

        Parameters
        ----------
        label : np.ndarray or torch.Tensor, shape (D, H, W) or (1, D, H, W)
            Raw segmentation map with integer values.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (3, D, H, W)
            Binary masks in channel order ``[WT, TC, ET]``.
        """
        if isinstance(label, torch.Tensor):
            return self._map_torch(label)
        return self._map_numpy(label)

    def to_binary_mask(
        self,
        label: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Alias for ``__call__``."""
        return self(label)

    @property
    def region_names(self) -> List[str]:
        """Ordered list of region names matching channel axis."""
        return self.config.region_names

    @property
    def dataset_name(self) -> str:
        """Name of the dataset this mapper is designed for."""
        return self.config.dataset_name

    # ------------------------------------------------------------------
    # NumPy implementation
    # ------------------------------------------------------------------

    def _map_numpy(self, label: np.ndarray) -> np.ndarray:
        """Convert raw numpy label map to (3, D, H, W) binary float32 array."""
        label = np.squeeze(label)  # remove leading channel dim if present
        channels = []
        for region in self.config.region_names:
            raw_vals = self.config.region_definitions[region]
            mask = np.zeros_like(label, dtype=np.float32)
            for v in raw_vals:
                mask = np.logical_or(mask, label == v).astype(np.float32)
            channels.append(mask)
        return np.stack(channels, axis=0)  # (3, D, H, W)

    # ------------------------------------------------------------------
    # PyTorch implementation
    # ------------------------------------------------------------------

    def _map_torch(self, label: torch.Tensor) -> torch.Tensor:
        """Convert raw torch label map to (3, D, H, W) binary float32 tensor."""
        label = label.squeeze()  # remove leading channel dim if present
        channels = []
        for region in self.config.region_names:
            raw_vals = self.config.region_definitions[region]
            mask = torch.zeros_like(label, dtype=torch.float32)
            for v in raw_vals:
                mask = torch.logical_or(mask.bool(), label == v).float()
            channels.append(mask)
        return torch.stack(channels, dim=0)  # (3, D, H, W)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_raw_label_name(self, value: int) -> str:
        """Return the human-readable name of a raw integer label value."""
        return self.config.raw_label_names.get(value, f"Unknown({value})")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset='{self.dataset_name}', "
            f"regions={self.region_names})"
        )


# ---------------------------------------------------------------------------
# Concrete mappers
# ---------------------------------------------------------------------------

class BraTS2021LabelMapper(_BaseLabelMapper):
    """
    Label mapper for the **BraTS 2021** glioma segmentation dataset.

    Raw labels
    ----------
    0  – Background
    1  – NCR  (Necrotic / Non-Enhancing Tumour Core)
    2  – ED   (Peritumoral Edema)
    3  – ET   (Enhancing Tumour)

    Output channels (index → region)
    ---------------------------------
    0 → WT  (Whole Tumour)    = labels {1, 2, 3}
    1 → TC  (Tumour Core)     = labels {1, 3}
    2 → ET  (Enhancing Tumour)= labels {3}

    Examples
    --------
    >>> import numpy as np
    >>> mapper = BraTS2021LabelMapper()
    >>> raw = np.random.randint(0, 4, size=(64, 64, 64))
    >>> masks = mapper(raw)          # numpy path
    >>> masks.shape
    (3, 64, 64, 64)
    >>> import torch
    >>> masks_t = mapper(torch.from_numpy(raw))   # torch path
    >>> masks_t.shape
    torch.Size([3, 64, 64, 64])
    """

    config: LabelConfig = BRATS2021_CONFIG


class BraTS2023METLabelMapper(_BaseLabelMapper):
    """
    Label mapper for the **BraTS 2023-MET** metastasis segmentation dataset.

    Raw labels
    ----------
    0  – Background
    1  – NETC (Non-Enhancing Tumour Core)
    2  – SNFH (Surrounding Non-Enhancing FLAIR Hyperintensity)
    3  – ET   (Enhancing Tumour)

    Output channels (index → region)
    ---------------------------------
    0 → WT  (Whole Tumour)    = labels {1, 2, 3}
    1 → TC  (Tumour Core)     = labels {1, 3}
    2 → ET  (Enhancing Tumour)= labels {3}

    Clinical note: SNFH (label 2) is conceptually similar to Edema in glioma
    but carries a different clinical interpretation in the metastasis context.

    Examples
    --------
    >>> import numpy as np
    >>> mapper = BraTS2023METLabelMapper()
    >>> raw = np.random.randint(0, 4, size=(64, 64, 64))
    >>> masks = mapper(raw)
    >>> masks.shape
    (3, 64, 64, 64)
    """

    config: LabelConfig = BRATS2023MET_CONFIG


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_MAPPER_REGISTRY: Dict[str, _BaseLabelMapper] = {
    "brats2021":     BraTS2021LabelMapper(),
    "brats21":       BraTS2021LabelMapper(),
    "glioma":        BraTS2021LabelMapper(),
    "brats2023met":  BraTS2023METLabelMapper(),
    "brats2023-met": BraTS2023METLabelMapper(),
    "met":           BraTS2023METLabelMapper(),
    "metastasis":    BraTS2023METLabelMapper(),
}


def get_label_mapper(dataset_name: str) -> _BaseLabelMapper:
    """
    Return the appropriate label mapper for *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        Case-insensitive key; accepted values:
        ``"brats2021"``, ``"brats21"``, ``"glioma"``,
        ``"brats2023met"``, ``"brats2023-met"``, ``"met"``, ``"metastasis"``.

    Returns
    -------
    BraTS2021LabelMapper or BraTS2023METLabelMapper

    Raises
    ------
    ValueError
        If *dataset_name* is not recognised.

    Examples
    --------
    >>> mapper = get_label_mapper("brats2021")
    >>> type(mapper)
    <class 'datasets.label_mapping.BraTS2021LabelMapper'>
    """
    key = dataset_name.lower().strip()
    if key not in _MAPPER_REGISTRY:
        valid = list(_MAPPER_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Valid options: {valid}"
        )
    mapper = _MAPPER_REGISTRY[key]
    logger.debug("Returning %s for dataset_name='%s'", type(mapper).__name__, dataset_name)
    return mapper


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def map_labels_to_regions(
    label_tensor: Union[np.ndarray, torch.Tensor],
    mapper: Optional[_BaseLabelMapper] = None,
    dataset_name: Optional[str] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a raw label map to a (3, D, H, W) binary mask using *mapper*.

    Exactly one of *mapper* or *dataset_name* must be supplied.

    Parameters
    ----------
    label_tensor : np.ndarray or torch.Tensor, shape (..., D, H, W)
        Raw integer segmentation map.
    mapper : _BaseLabelMapper, optional
        Pre-constructed mapper instance.
    dataset_name : str, optional
        Dataset key passed to :func:`get_label_mapper` to build a mapper
        on the fly.

    Returns
    -------
    np.ndarray or torch.Tensor, shape (3, D, H, W)
        Binary masks – channel 0: WT, channel 1: TC, channel 2: ET.

    Raises
    ------
    ValueError
        If neither or both of *mapper* / *dataset_name* are provided.

    Examples
    --------
    >>> import numpy as np
    >>> raw = np.zeros((64, 64, 64), dtype=np.int32)
    >>> raw[20:40, 20:40, 20:40] = 1
    >>> binary = map_labels_to_regions(raw, dataset_name="brats2021")
    >>> binary.shape
    (3, 64, 64, 64)
    """
    if mapper is None and dataset_name is None:
        raise ValueError("Provide either 'mapper' or 'dataset_name'.")
    if mapper is not None and dataset_name is not None:
        raise ValueError("Provide only one of 'mapper' or 'dataset_name', not both.")

    if mapper is None:
        mapper = get_label_mapper(dataset_name)  # type: ignore[arg-type]

    return mapper(label_tensor)


# ---------------------------------------------------------------------------
# Quick self-test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== BraTS 2021 Config ===")
    print(BRATS2021_CONFIG)
    print()
    print("=== BraTS 2023-MET Config ===")
    print(BRATS2023MET_CONFIG)
    print()

    rng = np.random.default_rng(0)
    raw_np = rng.integers(0, 4, size=(64, 64, 64), dtype=np.int32)
    raw_t  = torch.from_numpy(raw_np)

    for name in ["brats2021", "brats2023-met"]:
        mapper = get_label_mapper(name)
        out_np = mapper(raw_np)
        out_t  = mapper(raw_t)
        print(f"[{name}] numpy output shape : {out_np.shape}, dtype={out_np.dtype}")
        print(f"[{name}] torch output shape : {out_t.shape}, dtype={out_t.dtype}")
        for i, region in enumerate(mapper.region_names):
            print(f"  {region}: {int(out_np[i].sum())} positive voxels")
        print()

    # Test map_labels_to_regions convenience function
    binary = map_labels_to_regions(raw_np, dataset_name="brats2021")
    assert binary.shape == (3, 64, 64, 64), "Shape mismatch!"
    print("map_labels_to_regions: OK")
