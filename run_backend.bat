@echo off
title Multimodal MRI Brain Tumor Segmentation - Backend Server
echo =========================================================================
echo Starting Backend API Server (FastAPI + PyTorch)
echo Project: Multimodal MRI-Based Brain Tumor Segmentation Using an Improved Attention-Based Deep Learning Framework
echo =========================================================================
cd /d %~dp0
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
