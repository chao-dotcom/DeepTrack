"""Complete setup script for data and models"""
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("People Tracking System - Data & Models Setup")
    print("=" * 60)
    print()
    
    # Run model download
    print("Step 1: Setting up models...")
    print("-" * 60)
    try:
        from scripts.download_models import main as download_models
        download_models()
    except Exception as e:
        print(f"Error in model setup: {e}")
        print("You can run it manually: python scripts/download_models.py")
    
    print()
    print("Step 2: Setting up data directory...")
    print("-" * 60)
    try:
        from scripts.download_sample_data import main as setup_data
        setup_data()
    except Exception as e:
        print(f"Error in data setup: {e}")
        print("You can run it manually: python scripts/download_sample_data.py")
    
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Test with webcam: python -m src.inference.main --input 0 --display")
    print("2. Or add a video file to data/raw/ and process it")
    print("3. Start API: python -m src.api.main")
    print("4. Start UI: python -m src.ui.main")


if __name__ == "__main__":
    main()

