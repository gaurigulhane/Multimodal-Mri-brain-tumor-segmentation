"""
create_demo_samples.py
----------------------
Generates realistic synthetic 3D Multimodal MRI volumes (.nii.gz) for:
- T1 (T1-weighted)
- T1ce (T1 Contrast-Enhanced / Gadolinium)
- T2 (T2-weighted)
- FLAIR (Fluid Attenuated Inversion Recovery)
- Ground Truth Segmentation (NCR=1, ED=2, ET=3)

These can be used for end-to-end smoke testing, frontend upload testing,
and interactive visualization without requiring the full ~50GB BraTS datasets.
"""

import os
from pathlib import Path
import numpy as np
import nibabel as nib

def generate_synthetic_patient(output_dir: Path, patient_id: str = "BraTS2021_00001", shape=(64, 64, 64)):
    output_dir.mkdir(parents=True, exist_ok=True)
    D, H, W = shape
    
    # 3D coordinate grid centered at 0
    z, y, x = np.ogrid[-1:1:complex(0, D), -1:1:complex(0, H), -1:1:complex(0, W)]
    r_brain = np.sqrt(x**2 + y**2 + z**2)
    brain_mask = (r_brain < 0.85).astype(np.float32)
    
    # Brain structure (GM/WM gradients)
    base_brain = (1.0 - 0.5 * r_brain) * brain_mask
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, shape) * brain_mask
    
    # Tumor center offset
    cx, cy, cz = 0.25, 0.15, 0.0
    r_tumor = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    
    # Subregions:
    # Edema (outer shell): r < 0.35
    # Enhancing tumor (rim): 0.12 <= r < 0.22
    # Necrotic core (center): r < 0.12
    edema_mask = ((r_tumor < 0.35) & (r_tumor >= 0.22) & (brain_mask > 0)).astype(np.float32)
    et_mask = ((r_tumor < 0.22) & (r_tumor >= 0.12) & (brain_mask > 0)).astype(np.float32)
    ncr_mask = ((r_tumor < 0.12) & (brain_mask > 0)).astype(np.float32)
    
    # Construct Modalities:
    # 1. T1: normal brain has good contrast, tumor is hypo-intense
    t1 = base_brain * 0.7 - (edema_mask * 0.15 + et_mask * 0.1 + ncr_mask * 0.25) + noise
    t1 = np.clip(t1, 0, 1.0) * 1000.0
    
    # 2. T1ce: Enhancing tumor (ET) is hyper-intense (bright gadolinium uptake)
    t1ce = base_brain * 0.7 + et_mask * 0.75 - ncr_mask * 0.2 + noise
    t1ce = np.clip(t1ce, 0, 1.5) * 1000.0
    
    # 3. T2: Both edema and core are hyper-intense
    t2 = base_brain * 0.5 + edema_mask * 0.6 + et_mask * 0.4 + ncr_mask * 0.7 + noise
    t2 = np.clip(t2, 0, 1.5) * 1000.0
    
    # 4. FLAIR: Edema is markedly hyper-intense, CSF suppressed
    flair = base_brain * 0.6 + edema_mask * 0.8 + et_mask * 0.5 + ncr_mask * 0.3 + noise
    flair = np.clip(flair, 0, 1.5) * 1000.0
    
    # Ground truth segmentation mask: 0=Background, 1=NCR, 2=ED, 3=ET
    seg = np.zeros(shape, dtype=np.uint8)
    seg[ncr_mask > 0] = 1
    seg[edema_mask > 0] = 2
    seg[et_mask > 0] = 3
    
    # Affine matrix (1mm isotropic spacing)
    affine = np.eye(4)
    
    # Save as NIfTI
    files = {
        f"{patient_id}_t1.nii.gz": t1.astype(np.float32),
        f"{patient_id}_t1ce.nii.gz": t1ce.astype(np.float32),
        f"{patient_id}_t2.nii.gz": t2.astype(np.float32),
        f"{patient_id}_flair.nii.gz": flair.astype(np.float32),
        f"{patient_id}_seg.nii.gz": seg,
    }
    
    saved_paths = {}
    for filename, data in files.items():
        filepath = output_dir / filename
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(filepath))
        saved_paths[filename] = str(filepath)
        print(f"Generated: {filepath} ({data.shape})")
        
    return saved_paths

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "demo_samples"
    paths = generate_synthetic_patient(out_dir, "demo_patient_01", shape=(64, 64, 64))
    print(f"Successfully created 5 multimodal demo NIfTI volumes in {out_dir}")
