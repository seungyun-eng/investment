@echo off
cd /d "%~dp0..\.."
echo Starting dashboard server...
if exist ".venv\Scripts\python.exe" (
    start "Dashboard Server" ".venv\Scripts\python.exe" scripts\dashboard\run_dashboard.py
) else (
    start "Dashboard Server" python scripts\dashboard\run_dashboard.py
)
timeout /t 3 /nobreak >nul
start "" "http://localhost:8765"
