@echo off
:: ============================================================
::  Psychologist AI — Start all services (dev mode)
::  Runs backend + frontend in separate windows.
::  No Docker required.
:: ============================================================

echo.
echo  ==========================================
echo   Psychologist AI ^| Starting services...
echo  ==========================================
echo.

cd /d "%~dp0"

:: Activate venv and start backend in a new window
start "Psych AI - Backend" cmd /k "call .venv\Scripts\activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Start frontend in a new window
start "Psych AI - Frontend" cmd /k "cd frontend && npm run dev"

echo  Backend  : http://localhost:8000
echo  API docs : http://localhost:8000/docs
echo  Frontend : http://localhost:5173
echo.
echo  Both services are running in separate windows.
echo  Close those windows (or Ctrl+C inside them) to stop.
echo.
pause
