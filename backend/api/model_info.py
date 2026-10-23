"""
Model info endpoint.
GET /model/info - returns model architecture information.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


class ModelArchitectureComponent(BaseModel):
    name: str
    description: str
    parameters: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str
    in_channels: int
    out_channels: int
    input_modalities: List[str]
    output_regions: List[str]
    architecture_components: List[ModelArchitectureComponent]
    improvements_over_baseline: List[str]
    total_parameters: Optional[int]
    total_parameters_millions: Optional[float]
    checkpoint_loaded: bool
    checkpoint_path: Optional[str]


class BaselineModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str
    in_channels: int
    out_channels: int
    total_parameters: Optional[int]
    description: str


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the improved model architecture."""
    
    total_params = None
    try:
        from models.improved_attention_unet import ImprovedResAttentionUNet3D
        model = ImprovedResAttentionUNet3D(in_channels=4, out_channels=3)
        total_params = sum(p.numel() for p in model.parameters())
        del model
    except Exception:
        pass
    
    # Check for best checkpoint
    checkpoint_path = None
    checkpoint_loaded = False
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    best_ckpt = checkpoints_dir / "best_model.pth"
    if best_ckpt.exists():
        checkpoint_path = str(best_ckpt)
        checkpoint_loaded = True
    
    return ModelInfoResponse(
        model_name="Improved Multimodal Residual Attention U-Net",
        model_type="3D Encoder-Decoder with Attention",
        in_channels=4,
        out_channels=3,
        input_modalities=["T1", "T1ce (T1-Gd)", "T2", "FLAIR"],
        output_regions=["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"],
        architecture_components=[
            ModelArchitectureComponent(
                name="Multimodal Input Fusion",
                description="Four MRI modalities (T1, T1ce, T2, FLAIR) stacked as 4-channel input tensor",
                parameters="4 input channels"
            ),
            ModelArchitectureComponent(
                name="Residual Encoder",
                description="4-level encoder with 3D residual convolutional blocks (BN+ReLU+Conv+BN+ReLU+Conv + skip). Doubles features at each level.",
                parameters="Features: 32→64→128→256"
            ),
            ModelArchitectureComponent(
                name="Channel Attention (SE-Net)",
                description="Squeeze-and-Excitation channel attention at each encoder level. Recalibrates channel-wise feature responses.",
                parameters="Reduction ratio: 16"
            ),
            ModelArchitectureComponent(
                name="Spatial Attention",
                description="3D spatial attention using max+avg pooling followed by 7x7x7 convolution to generate spatial attention map.",
                parameters="Kernel: 7×7×7"
            ),
            ModelArchitectureComponent(
                name="Multi-Scale Bottleneck",
                description="Bottleneck block with parallel 3×3×3, dilated 3×3×3, and 1×1×1 branches for multi-scale feature extraction.",
                parameters="512 bottleneck features"
            ),
            ModelArchitectureComponent(
                name="Attention-Gated Skip Connections",
                description="Additive attention gates on skip connections suppress irrelevant background features before concatenation.",
                parameters="4 attention gates"
            ),
            ModelArchitectureComponent(
                name="Residual Decoder",
                description="4-level decoder with transposed convolution upsampling, attended skip connections, and residual blocks.",
                parameters="Features: 512→256→128→64→32"
            ),
            ModelArchitectureComponent(
                name="Segmentation Head",
                description="1×1×1 convolutional head producing 3-channel multilabel output (sigmoid activation for WT/TC/ET).",
                parameters="3 output channels"
            ),
        ],
        improvements_over_baseline=[
            "Residual skip connections in every conv block for improved gradient flow",
            "Channel attention (SE-Net style) for feature recalibration",
            "Spatial attention for localizing tumor regions",
            "Attention-gated skip connections to suppress background noise",
            "Multi-scale bottleneck block capturing features at multiple resolutions",
            "Multilabel sigmoid formulation supporting overlapping WT/TC/ET regions",
            "Dual-dataset support: BraTS 2021 (glioma) and BraTS 2023-MET (metastasis)",
            "Dataset-specific label adapters for clinically correct region definitions",
        ],
        total_parameters=total_params,
        total_parameters_millions=round(total_params / 1e6, 2) if total_params else None,
        checkpoint_loaded=checkpoint_loaded,
        checkpoint_path=checkpoint_path,
    )


@router.get("/baseline", response_model=BaselineModelInfoResponse)
async def get_baseline_model_info():
    """Get information about the baseline U-Net model."""
    
    total_params = None
    try:
        from models.baseline_model import UNet3D
        model = UNet3D(in_channels=4, out_channels=3)
        total_params = sum(p.numel() for p in model.parameters())
        del model
    except Exception:
        pass
    
    return BaselineModelInfoResponse(
        model_name="Baseline 3D U-Net",
        model_type="3D U-Net (Baseline)",
        in_channels=4,
        out_channels=3,
        total_parameters=total_params,
        description=(
            "Baseline 3D U-Net with double convolution blocks and skip connections. "
            "Accepts 4-channel multimodal MRI input (T1, T1ce, T2, FLAIR). "
            "Outputs 3-channel binary segmentation (WT, TC, ET). "
            "Used as reference for comparison with the improved architecture."
        ),
    )
