"""
Datasets package for Multimodal MRI Brain Tumor Segmentation.
"""

from datasets.brats2021_loader import BraTS2021Dataset, BraTS2021DataModule, create_synthetic_brats2021_sample
from datasets.brats2023_met_loader import BraTS2023METDataset, BraTS2023METDataModule, create_synthetic_brats2023met_sample
from datasets.label_mapping import (
    BraTS2021LabelMapper,
    BraTS2023METLabelMapper,
    get_label_mapper,
    BRATS2021_CONFIG,
    BRATS2023MET_CONFIG,
)
from datasets.preprocessing import MultimodalPreprocessor, get_monai_train_transforms, get_monai_val_transforms


def get_datamodule(dataset_name: str, data_root: str, **kwargs):
    """
    Factory to get appropriate DataModule.
    
    Args:
        dataset_name: 'brats2021' or 'brats2023_met'
        data_root: path to dataset root directory
        **kwargs: additional DataModule arguments
    
    Returns:
        DataModule instance
    """
    if dataset_name == 'brats2021':
        return BraTS2021DataModule(data_root=data_root, **kwargs)
    elif dataset_name == 'brats2023_met':
        return BraTS2023METDataModule(data_root=data_root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'brats2021' or 'brats2023_met'.")


__all__ = [
    'BraTS2021Dataset', 'BraTS2021DataModule', 'create_synthetic_brats2021_sample',
    'BraTS2023METDataset', 'BraTS2023METDataModule', 'create_synthetic_brats2023met_sample',
    'BraTS2021LabelMapper', 'BraTS2023METLabelMapper', 'get_label_mapper',
    'BRATS2021_CONFIG', 'BRATS2023MET_CONFIG',
    'MultimodalPreprocessor', 'get_monai_train_transforms', 'get_monai_val_transforms',
    'get_datamodule',
]
