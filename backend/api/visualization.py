"""
Visualization endpoint.
GET /visualization - returns visualization data for MRI slices.
"""

import sys
import json
import base64
import logging
from pathlib import Path
from typing import Optional, List
from io import BytesIO

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
router = APIRouter()


class SliceData(BaseModel):
    plane: str
    slice_index: int
    total_slices: int
    image_base64: Optional[str]
    modality: str


class VisualizationResponse(BaseModel):
    available: bool
    patient_id: Optional[str]
    modality: str
    plane: str
    slice_index: int
    total_slices: int
    data: Optional[str]  # base64 encoded PNG
    message: str


@router.get("", response_model=VisualizationResponse)
async def get_visualization(
    patient_id: str = Query(default="demo"),
    modality: str = Query(default="t1", description="t1, t1ce, t2, flair"),
    plane: str = Query(default="axial", description="axial, sagittal, coronal"),
    slice_index: int = Query(default=75, ge=0),
):
    """
    Get a visualization of an MRI slice.
    
    In production, this loads actual patient data from the dataset.
    For demonstration, returns a placeholder response.
    """
    
    # Check if we have any demo data available
    demo_data_dir = PROJECT_ROOT / "outputs" / "demo_slices"
    demo_file = demo_data_dir / f"{patient_id}_{modality}_{plane}_{slice_index}.png"
    
    if demo_file.exists():
        try:
            with open(demo_file, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            return VisualizationResponse(
                available=True,
                patient_id=patient_id,
                modality=modality,
                plane=plane,
                slice_index=slice_index,
                total_slices=155,
                data=img_data,
                message="Visualization loaded successfully.",
            )
        except Exception as e:
            logger.error(f"Failed to load visualization: {e}")
    
    # Try to generate a synthetic visualization for demonstration
    try:
        import numpy as np
        
        # Generate a brain-like synthetic slice for demo
        size = 240
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        r = np.sqrt(x**2 + y**2)
        brain_mask = (r < 0.85).astype(float)
        
        # Add some texture
        np.random.seed(slice_index + hash(modality) % 100)
        noise = np.random.randn(size, size) * 0.1
        intensity_map = (1 - r) * brain_mask + noise * brain_mask
        intensity_map = np.clip(intensity_map, 0, 1)
        
        # Different modalities have different contrast
        if modality == 't1ce':
            # T1ce has bright enhancement in tumor region
            tumor_mask = ((x - 0.2)**2 + (y - 0.1)**2 < 0.04).astype(float)
            intensity_map = intensity_map + tumor_mask * 0.5
            intensity_map = np.clip(intensity_map, 0, 1)
        elif modality == 't2' or modality == 'flair':
            # T2/FLAIR show edema as bright
            edema_mask = ((x - 0.2)**2 + (y - 0.1)**2 < 0.09).astype(float)
            intensity_map = intensity_map + edema_mask * 0.3
            intensity_map = np.clip(intensity_map, 0, 1)
        
        # Convert to uint8
        img_array = (intensity_map * 255).astype(np.uint8)
        
        # Convert to PNG using PIL if available
        try:
            from PIL import Image
            img = Image.fromarray(img_array, mode='L')
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return VisualizationResponse(
                available=True,
                patient_id=patient_id,
                modality=modality,
                plane=plane,
                slice_index=slice_index,
                total_slices=155,
                data=img_data,
                message="Synthetic demonstration image generated. Upload real NIfTI files for actual patient data.",
            )
        except ImportError:
            pass
    
    except ImportError:
        pass
    
    return VisualizationResponse(
        available=False,
        patient_id=patient_id,
        modality=modality,
        plane=plane,
        slice_index=slice_index,
        total_slices=155,
        data=None,
        message=(
            "No visualization data available. "
            "Upload NIfTI files via the segmentation page to generate patient-specific visualizations."
        ),
    )
