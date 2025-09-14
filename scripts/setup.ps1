Param(
  [switch]$Reinstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts/python.exe"

Write-Host "[setup] Project root: $projectRoot"

if ($Reinstall -and (Test-Path $venvPath)) {
  Write-Host "[setup] Reinstall requested: removing existing .venv"
  Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPython)) {
  Write-Host "[setup] Creating virtual environment at .venv"
  & py -3 -m venv $venvPath 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] 'py -3' not found, trying 'python'"
    & python -m venv $venvPath
  }
}

if (-not (Test-Path $venvPython)) {
  throw "Failed to create virtual environment. Ensure Python 3 is installed and on PATH."
}

Write-Host "[setup] Using interpreter: $venvPython"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "[setup] Verifying PaddleOCR optional ONNX backend availability"
try {
  & $venvPython -c "import onnxruntime; print('onnxruntime OK')"
}
catch { Write-Host "[setup] onnxruntime not available (optional)." }

Write-Host "[setup] Done. Interpreter pinned in .vscode/settings.json"

