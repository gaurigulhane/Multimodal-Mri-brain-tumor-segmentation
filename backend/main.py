"""
FastAPI Backend for Multimodal MRI-Based Brain Tumor Segmentation
Using an Improved Attention-Based Deep Learning Framework

This is the main application entry point. 
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn

from backend.api.health import router as health_router
from backend.api.predict import router as predict_router
from backend.api.metrics import router as metrics_router
from backend.api.model_info import router as model_info_router
from backend.api.datasets import router as datasets_router
from backend.api.visualization import router as visualization_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Multimodal MRI Brain Tumor Segmentation API",
    description=(
        "API for the project: Multimodal MRI-Based Brain Tumor Segmentation "
        "Using an Improved Attention-Based Deep Learning Framework. "
        "Supports BraTS 2021 and BraTS 2023-MET datasets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS - allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
app.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
app.include_router(model_info_router, prefix="/model", tags=["Model"])
app.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
app.include_router(visualization_router, prefix="/visualization", tags=["Visualization"])

# Serve outputs directory as static files if it exists
outputs_dir = PROJECT_ROOT / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Multimodal MRI Brain Tumor Segmentation Backend")
    logger.info("Improved Attention-Based Deep Learning Framework")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("API documentation available at: http://localhost:8000/docs")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("GPU not available — using CPU for inference")
    except ImportError:
        logger.warning("PyTorch not found — inference will not be available")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Backend server shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
