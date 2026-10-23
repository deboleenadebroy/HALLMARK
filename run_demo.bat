@echo off
REM Hallmark demo launcher for Windows.  Usage: double-click, or run  run_demo.bat  in a terminal.
cd /d "%~dp0"

where tesseract >nul 2>nul
if errorlevel 1 (
  if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    set "HALLMARK_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
  ) else (
    echo Tesseract OCR is not installed. Install it from https://github.com/UB-Mannheim/tesseract/wiki
    pause
    exit /b 1
  )
)

if not exist .venv (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat || (echo Could not create the Python environment. Install Python 3.10+ from python.org & pause & exit /b 1)
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo.
echo == Self-check: analysing every demo file live ==
python scripts\check_demo.py

if not defined HALLMARK_PORT set HALLMARK_PORT=8000
echo.
echo == Hallmark is starting. Open  http://127.0.0.1:%HALLMARK_PORT%  in your browser. Ctrl+C to stop. ==
python app.py
