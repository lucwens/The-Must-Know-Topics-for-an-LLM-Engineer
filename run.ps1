#!/usr/bin/env pwsh
# Windows PowerShell launcher. Mirrors run.sh.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install -q -r requirements.txt

$port = if ($env:PORT) { $env:PORT } else { "8000" }
& $py -m uvicorn server.main:app --host 0.0.0.0 --port $port
