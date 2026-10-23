"""
Prediction endpoint.
POST /predict - upload 4 NIfTI files and get tumor segmentation.
"""

import sys
import os
import tempfile
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
router = APIRouter()


class RegionStats(BaseModel):
    voxel_count: int
    volume_mm3: float
    percentage_of_brain: float


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    patient_id: str
    inference_time_seconds: float
    model_type: str
    model_loaded: bool
    checkpoint_used: Optional[str]
    warning: Optional[str]
    regions: Dict[str, RegionStats]
    output_shape: List[int]
    message: str


@router.post("", response_model=PredictionResponse)
async def predict_segmentation(
    t1_file: UploadFile = File(..., description="T1 MRI NIfTI file (.nii or .nii.gz)"),
    t1ce_file: UploadFile = File(..., description="T1ce MRI NIfTI file (.nii or .nii.gz)"),
    t2_file: UploadFile = File(..., description="T2 MRI NIfTI file (.nii or .nii.gz)"),
    flair_file: UploadFile = File(..., description="FLAIR MRI NIfTI file (.nii or .nii.gz)"),
    patient_id: str = Form(default="unknown"),
    model_type: str = Form(default="improved"),
):
    """
    Perform brain tumor segmentation on uploaded NIfTI files.
    
    Requires trained model checkpoint to produce real predictions.
    If no checkpoint is available, returns an informative error.
    """
    
    # Validate file extensions
    allowed_ext = {'.nii', '.gz'}
    for f in [t1_file, t1ce_file, t2_file, flair_file]:
        fname = f.filename or ""
        if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
            raise HTTPException(
                status_code=400,
                detail=f"File '{fname}' is not a valid NIfTI file. Please upload .nii or .nii.gz files."
            )
    
    # Save uploaded files to temp directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        t1_path = temp_dir / (t1_file.filename or "t1.nii.gz")
        t1ce_path = temp_dir / (t1ce_file.filename or "t1ce.nii.gz")
        t2_path = temp_dir / (t2_file.filename or "t2.nii.gz")
        flair_path = temp_dir / (flair_file.filename or "flair.nii.gz")
        
        for file_obj, save_path in [
            (t1_file, t1_path), (t1ce_file, t1ce_path),
            (t2_file, t2_path), (flair_file, flair_path)
        ]:
            content = await file_obj.read()
            with open(save_path, 'wb') as f:
                f.write(content)
        
        # Load predictor
        from inference.predict import TumorPredictor
        
        # Check for checkpoint
        checkpoint_path = None
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        best_ckpt = checkpoints_dir / "best_model.pth"
        if best_ckpt.exists():
            checkpoint_path = str(best_ckpt)
        
        predictor = TumorPredictor(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device='auto',
        )
        
        # Run inference
        result = predictor.predict(
            t1_path=str(t1_path),
            t1ce_path=str(t1ce_path),
            t2_path=str(t2_path),
            flair_path=str(flair_path),
        )
        
        # Compute region statistics
        regions_stats = {}
        prediction = result.get('prediction')  # shape (3, D, H, W)
        region_names = ['WT', 'TC', 'ET']
        
        if prediction is not None:
            import numpy as np
            total_brain_voxels = int(np.sum(prediction[0]) + 1)  # approximate
            for idx, region_name in enumerate(region_names):
                region_mask = prediction[idx]
                voxel_count = int(region_mask.sum())
                volume_mm3 = float(voxel_count * 1.0)  # assuming 1mm isotropic spacing
                pct = float(voxel_count / max(total_brain_voxels, 1) * 100)
                regions_stats[region_name] = RegionStats(
                    voxel_count=voxel_count,
                    volume_mm3=volume_mm3,
                    percentage_of_brain=round(pct, 2),
                )
        
        output_shape = list(result.get('prediction', [[]])[0].shape) if result.get('prediction') is not None else []
        
        return PredictionResponse(
            success=True,
            patient_id=patient_id,
            inference_time_seconds=round(result.get('inference_time_seconds', 0.0), 3),
            model_type=model_type,
            model_loaded=result.get('model_loaded', False),
            checkpoint_used=result.get('checkpoint_used'),
            warning=result.get('warning'),
            regions=regions_stats,
            output_shape=output_shape,
            message="Segmentation completed successfully." if result.get('model_loaded') else "Inference ran but no trained checkpoint was loaded. Results are not clinically meaningful.",
        )
    
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Required deep learning libraries not installed: {str(e)}. Please run: pip install torch monai nibabel"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    
    finally:
        # Clean up temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
