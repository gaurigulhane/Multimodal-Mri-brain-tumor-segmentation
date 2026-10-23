@echo off
title Multimodal MRI Brain Tumor Segmentation - Full System Launcher
echo =========================================================================
echo Launching Multimodal MRI Brain Tumor Segmentation Project
echo "Multimodal MRI-Based Brain Tumor Segmentation Using an Improved Attention-Based Deep Learning Framework"
echo =========================================================================
cd /d %~dp0

echo Starting Backend Server on http://127.0.0.1:8000 ...
start "Brain Tumor Segmentation - Backend" cmd /k "run_backend.bat"

echo Waiting 3 seconds for backend initialization...
timeout /t 3 /nobreak >nul

echo Starting Frontend Dev Server on http://127.0.0.1:5173 ...
start "Brain Tumor Segmentation - Frontend" cmd /k "run_frontend.bat"

echo.
echo Both servers started!
echo Frontend: http://localhost:5173
echo Backend API Docs: http://localhost:8000/docs
echo.
pause
