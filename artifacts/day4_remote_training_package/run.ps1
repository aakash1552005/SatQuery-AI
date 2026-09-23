# SatQuery AI -- Remote GPU Training and Evaluation Launcher (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "SatQuery AI -- Remote GPU Training & Evaluation Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not $env:SATQUERY_DATA_ROOT) {
    $env:SATQUERY_DATA_ROOT = Join-Path (Get-Location) "data"
}
Write-Host "Using SATQUERY_DATA_ROOT: $env:SATQUERY_DATA_ROOT" -ForegroundColor Green

# 1. Preflight Verification
Write-Host "`n[1/3] Running GPU Hardware & Manifest Preflight..." -ForegroundColor Yellow
python preflight.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Preflight failed. Check GPU configuration and dataset manifests."
    exit 1
}

# 2. Train LoRA Adapter
Write-Host "`n[2/3] Training SatQuery LoRA Adapter..." -ForegroundColor Yellow
python train.py `
    --config configs/bigearthnet_txt_lora.yaml `
    --train-manifest manifests/ben_train_subset.json `
    --val-manifest manifests/ben_val_subset.json `
    --output-dir checkpoints/run_001 `
    --seed 42
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed."
    exit 1
}

# 3. Evaluate Best Checkpoint
Write-Host "`n[3/3] Evaluating on Test Split..." -ForegroundColor Yellow
python evaluate.py `
    --checkpoint checkpoints/run_001/best_checkpoint `
    --manifest manifests/ben_test_subset.json `
    --output-dir evaluation_results
if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation failed."
    exit 1
}

Write-Host "`nExecution completed successfully! Results written to evaluation_results/" -ForegroundColor Green
