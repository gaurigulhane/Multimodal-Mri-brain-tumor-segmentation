"""
Backend API endpoint integration tests.
Tests /health, /model/info, /model/baseline, /metrics, /datasets, /visualization, and /predict.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import io
import nibabel as nib

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "improved" in data["project"].lower() or "multimodal" in data["project"].lower()


def test_model_info_endpoints():
    # Improved model info
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["in_channels"] == 4
    assert data["out_channels"] == 3
    assert "T1" in data["input_modalities"]
    assert "FLAIR" in data["input_modalities"]
    assert len(data["improvements_over_baseline"]) > 0

    # Baseline model info
    resp_base = client.get("/model/baseline")
    assert resp_base.status_code == 200
    base_data = resp_base.json()
    assert base_data["in_channels"] == 4
    assert base_data["out_channels"] == 3


def test_metrics_endpoints():
    response = client.get("/metrics?dataset=brats2021")
    assert response.status_code == 200
    data = response.json()
    assert "metrics_available" in data

    resp_hist = client.get("/metrics/training/history")
    assert resp_hist.status_code == 200
    hist_data = resp_hist.json()
    assert "available" in hist_data


def test_datasets_endpoints():
    response = client.get("/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) >= 2
    names = [d["short_name"] for d in data["datasets"]]
    assert "BraTS 2021" in names
    assert "BraTS 2023-MET" in names


def test_visualization_endpoint():
    response = client.get("/visualization?patient_id=demo&modality=flair&plane=axial&slice_index=75")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["data"] is not None  # base64 image
    assert data["plane"] == "axial"
    assert data["modality"] == "flair"


def test_predict_endpoint_validation():
    # Sending dummy text files instead of NIfTI should return 400
    files = {
        "t1_file": ("test.txt", io.BytesIO(b"dummy"), "text/plain"),
        "t1ce_file": ("test.txt", io.BytesIO(b"dummy"), "text/plain"),
        "t2_file": ("test.txt", io.BytesIO(b"dummy"), "text/plain"),
        "flair_file": ("test.txt", io.BytesIO(b"dummy"), "text/plain"),
    }
    response = client.post("/predict", files=files)
    assert response.status_code == 400
