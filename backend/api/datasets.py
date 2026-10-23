"""
Datasets info endpoint.
GET /datasets - returns information about supported datasets.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()


class DatasetLabelInfo(BaseModel):
    raw_label: int
    name: str
    abbreviation: str
    clinical_meaning: str


class DatasetRegionInfo(BaseModel):
    region_name: str
    abbreviation: str
    composed_of: List[str]
    clinical_significance: str


class DatasetInfo(BaseModel):
    name: str
    short_name: str
    year: int
    tumor_type: str
    modalities: List[str]
    modality_filenames: Dict[str, str]
    num_patients_total: int
    annotation_labels: List[DatasetLabelInfo]
    segmentation_regions: List[DatasetRegionInfo]
    label_differences_from_glioma: Optional[str]
    access_url: str
    download_requires_registration: bool
    registration_url: str
    citation: str


class DatasetsResponse(BaseModel):
    datasets: List[DatasetInfo]
    label_mapping_notes: str


@router.get("", response_model=DatasetsResponse)
async def get_datasets_info():
    """
    Get detailed information about the supported BraTS datasets,
    including label conventions and differences between them.
    """
    
    return DatasetsResponse(
        datasets=[
            DatasetInfo(
                name="Brain Tumor Segmentation Challenge 2021",
                short_name="BraTS 2021",
                year=2021,
                tumor_type="Glioma (High-Grade and Low-Grade)",
                modalities=["T1", "T1ce (T1-Gd)", "T2", "FLAIR"],
                modality_filenames={
                    "T1": "BraTS2021_XXXXX_t1.nii.gz",
                    "T1ce": "BraTS2021_XXXXX_t1ce.nii.gz",
                    "T2": "BraTS2021_XXXXX_t2.nii.gz",
                    "FLAIR": "BraTS2021_XXXXX_flair.nii.gz",
                    "Segmentation": "BraTS2021_XXXXX_seg.nii.gz",
                },
                num_patients_total=1251,
                annotation_labels=[
                    DatasetLabelInfo(raw_label=0, name="Background", abbreviation="BG", clinical_meaning="Non-tumor background tissue"),
                    DatasetLabelInfo(raw_label=1, name="Necrotic Tumor Core", abbreviation="NCR", clinical_meaning="Dead tumor cells, often seen in high-grade gliomas"),
                    DatasetLabelInfo(raw_label=2, name="Peritumoral Edematous/Invaded Tissue", abbreviation="ED", clinical_meaning="Swelling and infiltrated tissue surrounding the tumor"),
                    DatasetLabelInfo(raw_label=3, name="GD-Enhancing Tumor", abbreviation="ET", clinical_meaning="Active tumor cells that enhance with gadolinium contrast"),
                ],
                segmentation_regions=[
                    DatasetRegionInfo(
                        region_name="Whole Tumor",
                        abbreviation="WT",
                        composed_of=["NCR (1)", "ED (2)", "ET (3)"],
                        clinical_significance="Total tumor extent including all sub-regions",
                    ),
                    DatasetRegionInfo(
                        region_name="Tumor Core",
                        abbreviation="TC",
                        composed_of=["NCR (1)", "ET (3)"],
                        clinical_significance="Core tumor mass excluding edema, relevant for surgical planning",
                    ),
                    DatasetRegionInfo(
                        region_name="Enhancing Tumor",
                        abbreviation="ET",
                        composed_of=["ET (3)"],
                        clinical_significance="Active tumor cells, important for treatment response monitoring",
                    ),
                ],
                label_differences_from_glioma=None,
                access_url="https://www.synapse.org/Synapse:syn25829067",
                download_requires_registration=True,
                registration_url="https://www.synapse.org/",
                citation="Baid et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv 2021",
            ),
            DatasetInfo(
                name="Brain Tumor Segmentation Challenge 2023 - Brain Metastases",
                short_name="BraTS 2023-MET",
                year=2023,
                tumor_type="Brain Metastases (from various primary cancers)",
                modalities=["T1n (native T1)", "T1c (T1 post-contrast)", "T2w", "T2f (FLAIR)"],
                modality_filenames={
                    "T1n": "ASNR-MICCAI-BraTS2023-MET-XXXX-000_t1n.nii.gz",
                    "T1c": "ASNR-MICCAI-BraTS2023-MET-XXXX-000_t1c.nii.gz",
                    "T2w": "ASNR-MICCAI-BraTS2023-MET-XXXX-000_t2w.nii.gz",
                    "T2f": "ASNR-MICCAI-BraTS2023-MET-XXXX-000_t2f.nii.gz",
                    "Segmentation": "ASNR-MICCAI-BraTS2023-MET-XXXX-000_seg.nii.gz",
                },
                num_patients_total=238,
                annotation_labels=[
                    DatasetLabelInfo(raw_label=0, name="Background", abbreviation="BG", clinical_meaning="Non-tumor background tissue"),
                    DatasetLabelInfo(raw_label=1, name="Non-Enhancing Tumor Core", abbreviation="NETC", clinical_meaning="Non-enhancing core of metastatic tumor (distinct from glioma NCR)"),
                    DatasetLabelInfo(raw_label=2, name="Surrounding Non-Enhancing FLAIR Hyperintensity", abbreviation="SNFH", clinical_meaning="Peritumoral FLAIR signal changes (may include edema and infiltration; distinct clinical meaning from glioma ED)"),
                    DatasetLabelInfo(raw_label=3, name="Enhancing Tumor", abbreviation="ET", clinical_meaning="Contrast-enhancing metastatic tumor — most metastases predominantly enhance"),
                ],
                segmentation_regions=[
                    DatasetRegionInfo(
                        region_name="Whole Tumor",
                        abbreviation="WT",
                        composed_of=["NETC (1)", "SNFH (2)", "ET (3)"],
                        clinical_significance="Total metastatic lesion extent",
                    ),
                    DatasetRegionInfo(
                        region_name="Tumor Core",
                        abbreviation="TC",
                        composed_of=["NETC (1)", "ET (3)"],
                        clinical_significance="Core lesion mass used for stereotactic radiosurgery planning",
                    ),
                    DatasetRegionInfo(
                        region_name="Enhancing Tumor",
                        abbreviation="ET",
                        composed_of=["ET (3)"],
                        clinical_significance="Primary tumor component in metastases; most metastases are predominantly ET",
                    ),
                ],
                label_differences_from_glioma=(
                    "IMPORTANT: Although label numbers 1, 2, 3 are shared, the clinical entities differ. "
                    "Label 1 in MET is NETC (not NCR); Label 2 is SNFH (not ED — different clinical significance). "
                    "SNFH in metastases represents peritumoral signal changes but is clinically distinct from glioma edema. "
                    "This project uses a dataset-specific label adapter to correctly name and handle these differences. "
                    "The region definitions WT=1+2+3, TC=1+3, ET=3 are structurally identical but clinically different."
                ),
                access_url="https://www.synapse.org/Synapse:syn51156910",
                download_requires_registration=True,
                registration_url="https://www.synapse.org/",
                citation="Moawad et al., The Brain Tumor Segmentation (BraTS) Challenge 2023: Brain MR Image Segmentation for Brain Metastases, arXiv 2023",
            ),
        ],
        label_mapping_notes=(
            "Both datasets use label values 0, 1, 2, 3 but with different clinical meanings. "
            "This project implements separate LabelMapper classes for each dataset. "
            "All metrics are computed independently per dataset using dataset-specific region definitions. "
            "Results from BraTS 2021 and BraTS 2023-MET are never merged without clearly documenting the difference."
        ),
    )
