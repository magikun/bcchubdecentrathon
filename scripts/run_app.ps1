$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  Write-Host "[run] .venv not found. Bootstrapping..."
  & (Join-Path $PSScriptRoot "setup.ps1")
}

Write-Host "[run] Starting Streamlit app with project venv"
& $venvPython -m streamlit run (Join-Path $projectRoot "app.py")

