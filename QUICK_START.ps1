# Quick Start Script for ML Training and Inference
# Run this to train model and test inference pipeline
# Perfect for hackathon beginners!

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  GHOSTLOAD MAPPER - ML QUICK START" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Check environment
Write-Host "[Step 1/3] Checking environment..." -ForegroundColor Yellow
if (-not (Test-Path "machine_learning\venv\Scripts\python.exe")) {
    Write-Host "  ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "  Please run: python -m venv machine_learning\venv" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Virtual environment found" -ForegroundColor Green

# Step 2: Train model
Write-Host "`n[Step 2/3] Training ML model..." -ForegroundColor Yellow
Write-Host "  This will take 2-3 minutes...`n" -ForegroundColor Gray

& machine_learning\venv\Scripts\python.exe machine_learning\train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n  ERROR: Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n  ✓ Model trained successfully!" -ForegroundColor Green

# Step 3: Test inference
Write-Host "`n[Step 3/3] Testing inference pipeline..." -ForegroundColor Yellow

& machine_learning\venv\Scripts\python.exe machine_learning\pipeline\inference_pipeline.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n  ERROR: Inference test failed!" -ForegroundColor Red
    exit 1
}

# Success!
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ✓ ALL DONE! ML SYSTEM READY!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Share with backend team:" -ForegroundColor White
Write-Host "   machine_learning\pipeline\inference_pipeline.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Backend integration code:" -ForegroundColor White
Write-Host "   from machine_learning.pipeline.inference_pipeline import predict_meter_risk" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Train again anytime with:" -ForegroundColor White
Write-Host "   python machine_learning\train.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. See guides in machine_learning\docs" -ForegroundColor Cyan
Write-Host ""
