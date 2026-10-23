"""
Metrics endpoint.
GET /metrics - returns evaluation metrics from stored results.
GET /training/history - returns training history.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


class RegionMetrics(BaseModel):
    dice: Optional[float] = None
    iou: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    hd95: Optional[float] = None


class DatasetMetrics(BaseModel):
    dataset_name: str
    num_patients: Optional[int] = None
    WT: Optional[RegionMetrics] = None
    TC: Optional[RegionMetrics] = None
    ET: Optional[RegionMetrics] = None
    mean_dice: Optional[float] = None
    evaluation_status: str


class MetricsResponse(BaseModel):
    improved_model: Optional[DatasetMetrics] = None
    baseline_model: Optional[DatasetMetrics] = None
    metrics_available: bool
    message: str


class TrainingHistoryResponse(BaseModel):
    available: bool
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]
    val_dice_wt: List[float]
    val_dice_tc: List[float]
    val_dice_et: List[float]
    val_mean_dice: List[float]
    message: str


def _load_metrics_file(path: Path) -> Optional[Dict]:
    """Load a JSON metrics file if it exists."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


@router.get("", response_model=MetricsResponse)
async def get_metrics(dataset: str = "brats2021"):
    """
    Get evaluation metrics for the models.
    Returns actual computed metrics if evaluation has been run,
    otherwise returns status indicating training/evaluation is needed.
    """
    outputs_dir = PROJECT_ROOT / "outputs" / "metrics"
    
    # Try to load evaluation results
    improved_path = outputs_dir / f"improved_{dataset}_results.json"
    baseline_path = outputs_dir / f"baseline_{dataset}_results.json"
    
    improved_data = _load_metrics_file(improved_path)
    baseline_data = _load_metrics_file(baseline_path)
    
    def _parse_dataset_metrics(data: Optional[Dict], name: str, ds: str) -> DatasetMetrics:
        if data is None:
            return DatasetMetrics(
                dataset_name=ds,
                evaluation_status="not_evaluated",
            )
        
        mean_metrics = data.get("mean_metrics", {})
        
        def _region(r: str) -> Optional[RegionMetrics]:
            rm = mean_metrics.get(r, {})
            if not rm:
                return None
            return RegionMetrics(
                dice=rm.get("dice"),
                iou=rm.get("iou"),
                precision=rm.get("precision"),
                recall=rm.get("recall"),
                specificity=rm.get("specificity"),
                hd95=rm.get("hd95"),
            )
        
        return DatasetMetrics(
            dataset_name=ds,
            num_patients=data.get("num_patients"),
            WT=_region("WT"),
            TC=_region("TC"),
            ET=_region("ET"),
            mean_dice=mean_metrics.get("mean_dice"),
            evaluation_status="evaluated",
        )
    
    metrics_available = improved_data is not None or baseline_data is not None
    
    message = (
        "Metrics loaded from evaluation results." if metrics_available
        else (
            "No evaluation results found. To generate metrics: "
            "(1) Obtain BraTS dataset, (2) Train the model with training/train.py, "
            "(3) Run evaluation/evaluate.py. "
            "Metrics will be saved to outputs/metrics/ automatically."
        )
    )
    
    return MetricsResponse(
        improved_model=_parse_dataset_metrics(improved_data, "improved", dataset),
        baseline_model=_parse_dataset_metrics(baseline_data, "baseline", dataset),
        metrics_available=metrics_available,
        message=message,
    )


@router.get("/training/history", response_model=TrainingHistoryResponse)
async def get_training_history():
    """Get training history (loss and metric curves)."""
    history_path = PROJECT_ROOT / "outputs" / "training_history.json"
    
    if not history_path.exists():
        return TrainingHistoryResponse(
            available=False,
            epochs=[],
            train_loss=[],
            val_loss=[],
            val_dice_wt=[],
            val_dice_tc=[],
            val_dice_et=[],
            val_mean_dice=[],
            message=(
                "No training history found. Train the model using: "
                "python training/train.py --config training/config.yaml"
            ),
        )
    
    try:
        with open(history_path) as f:
            history = json.load(f)
        
        return TrainingHistoryResponse(
            available=True,
            epochs=history.get("epochs", []),
            train_loss=history.get("train_loss", []),
            val_loss=history.get("val_loss", []),
            val_dice_wt=history.get("val_dice_wt", []),
            val_dice_tc=history.get("val_dice_tc", []),
            val_dice_et=history.get("val_dice_et", []),
            val_mean_dice=history.get("val_mean_dice", []),
            message="Training history loaded successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load training history: {e}")
