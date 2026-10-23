@echo off
echo =======================================================
echo Pushing project to GitHub repository:
echo https://github.com/gaurigulhane/Multimodal-Mri-brain-tumor-segmentation
echo =======================================================
cd /d %~dp0
git push -u origin main
echo.
pause
