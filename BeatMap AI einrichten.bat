@echo off
cd /d "%~dp0"
rem First-time setup: detects NVIDIA / AMD / no graphics card and installs the matching
rem PyTorch into its own environment (.venv-cuda / .venv-rocm / .venv-cpu).
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -c "" >nul 2>nul && (
        py -3.12 scripts\setup_env.py %*
        goto done
    )
    py -3 scripts\setup_env.py %*
    goto done
)
python scripts\setup_env.py %*
:done
pause
