"""
Models package for Multimodal MRI Brain Tumor Segmentation.
"""

from models.improved_attention_unet import ImprovedResAttentionUNet3D, get_improved_model
from models.baseline_model import UNet3D, get_baseline_model


def get_model(model_type: str = 'improved', **kwargs):
    """
    Factory function to get model by type.
    
    Args:
        model_type: 'improved' or 'baseline'
        **kwargs: additional model arguments
    
    Returns:
        Model instance
    """
    if model_type == 'improved':
        return get_improved_model(**kwargs)
    elif model_type == 'baseline':
        return get_baseline_model(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'improved' or 'baseline'.")


__all__ = [
    'ImprovedResAttentionUNet3D',
    'UNet3D',
    'get_improved_model',
    'get_baseline_model',
    'get_model',
]
