#!/bin/bash
# Setup script for People Tracking System

echo "Setting up People Tracking System..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed data/annotations
mkdir -p models/checkpoints models/exports models/optimized
mkdir -p logs

# Copy example config if config doesn't exist
if [ ! -f "configs/deployment/config.yaml" ]; then
    echo "Creating config file from example..."
    cp configs/deployment/config.example.yaml configs/deployment/config.yaml
fi

echo "Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the API server, run: python -m src.api.main"
echo "To start the UI, run: python -m src.ui.main"

