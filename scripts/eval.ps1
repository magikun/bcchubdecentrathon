$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv/\Scripts/python.exe"

if (-not (Test-Path $python)) {
  Write-Host "[eval] .venv missing, bootstrapping..."
  & (Join-Path $PSScriptRoot "setup.ps1")
}

Write-Host "[eval] Exporting PDF first pages to eval_images/"
& $python -m src.eval.export_pdf_images --pdf_dir "хакатон" --out "eval_images"

Write-Host "[eval] Building GT JSONs from XLSX into eval_gts/"
& $python -m src.eval.build_gt_from_xlsx --xlsx_dir "хакатон" --out "eval_gts"

Write-Host "[eval] Running evaluation (PaddleOCR vs Tesseract)"
& $python -m src.eval.run_eval --images "eval_images" --gts "eval_gts" --engine tesseract --out "eval_results.json"

Write-Host "[eval] Aggregating results → eval_summary.json"
& $python -m src.eval.aggregate --results "eval_results.json" | Set-Content -Encoding utf8 (Join-Path $projectRoot "eval_summary.json")

Write-Host "[eval] Done. See eval_results.json and eval_summary.json"


