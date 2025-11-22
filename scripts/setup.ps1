# Setup script for People Tracking System (PowerShell)

Write-Host "Setting up People Tracking System..." -ForegroundColor Green

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw", "data\processed", "data\annotations" | Out-Null
New-Item -ItemType Directory -Force -Path "models\checkpoints", "models\exports", "models\optimized" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Copy example config if config doesn't exist
if (-not (Test-Path "configs\deployment\config.yaml")) {
    Write-Host "Creating config file from example..." -ForegroundColor Yellow
    Copy-Item "configs\deployment\config.example.yaml" "configs\deployment\config.yaml"
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To activate the virtual environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To start the API server, run: python -m src.api.main" -ForegroundColor Cyan
Write-Host "To start the UI, run: python -m src.ui.main" -ForegroundColor Cyan

