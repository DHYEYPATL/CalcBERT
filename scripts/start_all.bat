@echo off
REM Start all services for CalcBERT demo (Windows version)
REM Usage: scripts\start_all.bat

echo ============================================================
echo Starting CalcBERT Services
echo ============================================================
echo.

REM Start backend in new window
echo Starting Backend (FastAPI)...
start "CalcBERT Backend" cmd /k "cd /d %~dp0.. && py -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload"
echo   Backend starting at http://127.0.0.1:8000
echo   API Docs at http://127.0.0.1:8000/docs
echo.

REM Wait for backend to be ready
echo Waiting for backend to be ready...
timeout /t 5 /nobreak >nul

REM Check if UI exists
if exist "%~dp0..\ui\app.py" (
    echo Starting UI (Streamlit)...
    start "CalcBERT UI" cmd /k "cd /d %~dp0..\ui && streamlit run app.py --server.port 8501"
    echo   UI starting at http://localhost:8501
) else (
    echo UI not found (ui\app.py), skipping...
)

echo.
echo ============================================================
echo All services started in separate windows!
echo ============================================================
echo Backend API: http://127.0.0.1:8000
echo API Docs: http://127.0.0.1:8000/docs
if exist "%~dp0..\ui\app.py" (
    echo UI: http://localhost:8501
)
echo.
echo Close the terminal windows to stop services
echo ============================================================
echo.
pause
