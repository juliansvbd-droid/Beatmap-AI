@echo off
cd /d "%~dp0"
rem Prefer the ROCm environment (AMD GPU training); fall back to the DirectML one.
if exist ".venv-rocm\Scripts\pythonw.exe" (
    start "" ".venv-rocm\Scripts\pythonw.exe" -m beatmap_ai ui %*
    exit /b 0
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m beatmap_ai ui %*
    exit /b 0
)
echo Die Projektumgebung wurde nicht gefunden.
echo Bitte richte zuerst die Python-Umgebung fuer dieses Projekt ein.
pause
