"""Script to download pre-trained models for people tracking"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_yolo_model():
    """Download YOLOv8 model for person detection"""
    try:
        from ultralytics import YOLO
        print("Downloading YOLOv8 model...")
        
        # Create models directory
        model_dir = Path("models/checkpoints")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download YOLOv8n (nano - smallest, fastest)
        # Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
        model_name = 'yolov8n.pt'
        print(f"Downloading {model_name}...")
        model = YOLO(model_name)  # This will auto-download if not present
        
        # Save to our models directory
        model_path = model_dir / model_name
        if not model_path.exists():
            import shutil
            # Find where ultralytics saved it (usually in ~/.ultralytics)
            from ultralytics.utils import SETTINGS
            ultralytics_dir = Path(SETTINGS['weights_dir'])
            source_path = ultralytics_dir / model_name
            if source_path.exists():
                shutil.copy(source_path, model_path)
                print(f"Model saved to: {model_path}")
            else:
                # Model is in memory, save it
                model.save(model_path)
                print(f"Model saved to: {model_path}")
        else:
            print(f"Model already exists at: {model_path}")
        
        print(f"✓ YOLOv8 model downloaded successfully!")
        return str(model_path)
        
    except ImportError:
        print("Error: ultralytics not installed. Installing...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        return download_yolo_model()
    except Exception as e:
        print(f"Error downloading YOLOv8 model: {e}")
        print("\nAlternative: You can manually download YOLOv8 from:")
        print("https://github.com/ultralytics/ultralytics")
        return None


def download_deepsort_weights():
    """Download DeepSORT weights for tracking"""
    import urllib.request
    
    model_dir = Path("models/checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # DeepSORT weights URL
    weights_url = "https://drive.google.com/uc?id=1_qwTWdzT9dWNudpusgKavj_4elGgbKoUN"
    weights_path = model_dir / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"
    
    print("Note: DeepSORT weights need to be downloaded manually.")
    print("Visit: https://github.com/ZQPei/deep_sort_pytorch")
    print("Or use a simpler tracker like ByteTrack or CentroidTracker")
    
    return None


def main():
    """Main function to download all models"""
    print("=" * 60)
    print("Downloading Pre-trained Models for People Tracking")
    print("=" * 60)
    print()
    
    # Download YOLOv8
    yolo_path = download_yolo_model()
    
    print()
    print("=" * 60)
    print("Model Download Complete!")
    print("=" * 60)
    print(f"\nYOLOv8 model: {yolo_path or 'Not downloaded'}")
    print("\nYou can now use the model for people detection.")
    print("Example: python -m src.inference.main --input 0 --model models/checkpoints/yolov8n.pt")


if __name__ == "__main__":
    main()

