@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if "%PORT%"=="" set PORT=8000
python -m uvicorn server.main:app --host 0.0.0.0 --port %PORT%
