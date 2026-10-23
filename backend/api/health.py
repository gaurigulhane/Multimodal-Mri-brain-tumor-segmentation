"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
import torch
import platform
import sys
from datetime import datetime, timezone

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    project: str
    version: str
    python_version: str
    pytorch_available: bool
    pytorch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    platform: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and system capabilities."""
    
    pytorch_available = True
    pytorch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = None
    
    try:
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pytorch_available = False
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        project="Multimodal MRI-Based Brain Tumor Segmentation Using an Improved Attention-Based Deep Learning Framework",
        version="1.0.0",
        python_version=sys.version,
        pytorch_available=pytorch_available,
        pytorch_version=pytorch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        platform=platform.platform(),
    )
