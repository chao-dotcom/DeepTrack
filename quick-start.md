# Quick Start Guide - People Tracking System

Get up and running with the People Tracking System in minutes.

## Prerequisites

- Python 3.8+
- pip
- (Optional) GPU with CUDA for faster processing

## Installation

### 1. Clone and Setup

```bash
# Navigate to project directory
cd people-tracking-system

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate.bat
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download YOLOv8 detection model (auto-downloads if not present)
python scripts/download_models.py
```

The model will be saved to `models/checkpoints/yolov8n.pt`.

## Quick Test

### Test with Webcam

```bash
python -m src.inference.main --input 0 --display
```

Press `q` to quit.

### Test with Video File

```bash
python -m src.inference.main --input path/to/video.mp4 --output output.mp4
```

## Common Usage

### 1. Run Tracking on Video

```bash
python -m src.inference.main \
    --input data/raw/video.mp4 \
    --output data/processed/output.mp4 \
    --config configs/tracking_config.yaml
```

### 2. Start API Server

```bash
python -m src.api.main
```

Then visit `http://localhost:8000/docs` for API documentation.

### 3. Start Web UI

```bash
python -m src.ui.main
```

Then visit `http://localhost:8501` in your browser.

## Configuration

Edit `configs/tracking_config.yaml` to adjust:

- Detection threshold (`detection.conf_threshold`)
- Tracking parameters (`tracking.max_dist`, `tracking.max_age`)
- Model paths

## Example: Process MOT20 Sequence

```bash
# Process a MOT20 sequence
python -m src.inference.main \
    --input data/raw/MOT20/MOT20/train/MOT20-01/img1 \
    --output data/processed/MOT20-01_tracked.mp4 \
    --config configs/tracking_config.yaml
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`

### GPU Not Detected

The system works on CPU but is slower. For GPU:
1. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
2. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Model Download Issues

Models auto-download on first use. If issues:
- Check internet connection
- Manually download from Ultralytics: https://github.com/ultralytics/ultralytics

## Next Steps

- **Training**: See `src/training/` for model training scripts
- **Evaluation**: Run `python scripts/evaluate_benchmark.py` for metrics
- **Deployment**: See `docker-compose.yml` for Docker setup
- **Documentation**: Check `docs-zh/` for detailed Chinese documentation

## Need Help?

- Check `README.md` for full documentation
- Review `configs/tracking_config.yaml` for configuration options
- See `src/inference/main.py` for CLI options

---

**Ready to track!** 🚀

