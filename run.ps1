$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path .venv)) {
    python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
pip install -q -r requirements.txt
$port = if ($env:PORT) { $env:PORT } else { "8000" }
python -m uvicorn server.main:app --host 0.0.0.0 --port $port
