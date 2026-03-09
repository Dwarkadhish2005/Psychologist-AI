@echo off
:: Stops both services started by start.bat
echo Stopping Psychologist AI services...
taskkill /FI "WINDOWTITLE eq Psych AI - Backend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Psych AI - Frontend*" /T /F >nul 2>&1
:: Also kill any lingering uvicorn / vite processes on those ports
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 "') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo Done.
pause
