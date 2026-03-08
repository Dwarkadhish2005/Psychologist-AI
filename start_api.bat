@echo off
echo ================================================
echo  Psychologist AI - Phase 6 Launcher
echo ================================================
echo.
echo Starting FastAPI backend on http://localhost:8000
echo Swagger docs: http://localhost:8000/docs
echo.
cd /d "%~dp0"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload