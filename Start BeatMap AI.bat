@echo off
cd /d "%~dp0"
rem Use the environment "BeatMap AI einrichten.bat" created: AMD (ROCm), NVIDIA (CUDA),
rem processor only, or the old DirectML one. In the app, "Rechnen auf" picks GPU or CPU.
for %%V in (.venv-rocm .venv-cuda .venv-cpu .venv) do (
    if exist "%%V\Scripts\pythonw.exe" (
        start "" "%%V\Scripts\pythonw.exe" -m beatmap_ai ui %*
        exit /b 0
    )
)
echo Die Projektumgebung wurde nicht gefunden.
echo Bitte zuerst "BeatMap AI einrichten.bat" ausfuehren.
pause
