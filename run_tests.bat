@echo off
title Multimodal MRI Brain Tumor Segmentation - Test Suite
echo =========================================================================
echo Running Full Test Suite (Models, Pipeline, Metrics, API Endpoints)
echo =========================================================================
cd /d %~dp0
python -m pytest tests/ -v
echo.
pause
