@echo off
setlocal

cd /d "%~dp0"

echo [Sharky AI Backend] Checking for Python virtual environment...

if not exist venv (
    echo [Sharky AI Backend] Creating virtual environment...
    python -m venv venv
)

echo [Sharky AI Backend] Activating virtual environment...
call venv\Scripts\activate.bat

echo [Sharky AI Backend] Installing requirements...
pip install -r requirements.txt

echo [Sharky AI Backend] Starting AI Server on http://127.0.0.1:8000
echo You can keep this window open while using the SharkyExtension.
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause
