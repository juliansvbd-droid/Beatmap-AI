@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~dp0.venv-rocm\Scripts\python.exe' -ArgumentList '-u','%~dp0scripts\night_run.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden"
endlocal
