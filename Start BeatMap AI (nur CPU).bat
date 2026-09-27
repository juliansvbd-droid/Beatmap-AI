@echo off
cd /d "%~dp0"
rem Same app, but without the graphics card: for testing while a training job uses the GPU.
rem Slower (a few minutes per difficulty), but it cannot take GPU memory from the training.
set HIP_VISIBLE_DEVICES=
set CUDA_VISIBLE_DEVICES=
set HIP_VISIBLE_DEVICES=-1
set CUDA_VISIBLE_DEVICES=-1
if exist ".venv-rocm\Scripts\pythonw.exe" (
    start "" ".venv-rocm\Scripts\pythonw.exe" -m beatmap_ai ui %*
    exit /b 0
)
echo Die Projektumgebung wurde nicht gefunden.
pause
