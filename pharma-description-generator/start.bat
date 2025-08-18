@echo off
echo ============================================================
echo   PHARMACEUTICAL DESCRIPTION GENERATOR - SETUP SCRIPT
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "output" mkdir output

echo.
echo [3/3] Starting the application...
echo.
echo ============================================================
echo   APPLICATION READY!
echo ============================================================
echo   Open your browser and go to: http://127.0.0.1:5000
echo ============================================================
echo.

python app.py
