"""Setup script to get the system up and running"""
import subprocess
import sys
from pathlib import Path
import os


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'models/checkpoints',
        'outputs',
        'uploads',
        'checkpoints'
    ]
    
    print("\nCreating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✓ Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if requirements.txt exists
    if not Path('requirements.txt').exists():
        print("✗ requirements.txt not found")
        return False
    
    # Install base requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing base dependencies"
    )
    
    if not success:
        return False
    
    # Install optional dependencies (with error handling)
    optional_deps = [
        "motmetrics",  # For MOT evaluation
        "scikit-learn",  # For clustering
        "redis",  # For job queue
        "celery",  # For background tasks
        "onnxruntime",  # For ONNX inference
    ]
    
    print("\nInstalling optional dependencies...")
    for dep in optional_deps:
        try:
            subprocess.run(
                f"{sys.executable} -m pip install {dep}",
                shell=True,
                check=True,
                capture_output=True
            )
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠ Could not install {dep} (optional, continuing...)")
    
    return True


def download_yolo_model():
    """Download YOLOv8 model"""
    print("\nDownloading YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        
        model_path = Path('models/checkpoints/yolov8n.pt')
        if model_path.exists():
            print(f"✓ Model already exists at {model_path}")
            return True
        
        print("Downloading YOLOv8n (nano) model...")
        model = YOLO('yolov8n.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to checkpoints if downloaded elsewhere
        import shutil
        if Path('yolov8n.pt').exists():
            shutil.move('yolov8n.pt', str(model_path))
        
        print(f"✓ Model downloaded to {model_path}")
        return True
    except Exception as e:
        print(f"⚠ Could not download YOLOv8 model: {e}")
        print("You can download it manually or it will auto-download on first use")
        return False


def verify_installation():
    """Verify installation"""
    print("\nVerifying installation...")
    
    checks = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "Ultralytics YOLO"),
        ("fastapi", "FastAPI"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for module, name in checks:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not found")
            all_ok = False
    
    return all_ok


def main():
    """Main setup function"""
    print("="*60)
    print("People Tracking System - Setup")
    print("="*60)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n✗ Setup failed: Python version too old")
        sys.exit(1)
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed: Could not install dependencies")
        print("Try installing manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 4: Verify installation
    if not verify_installation():
        print("\n⚠ Some dependencies are missing, but setup can continue")
    
    # Step 5: Download YOLO model (optional)
    download_yolo_model()
    
    # Final message
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the system:")
    print("   python -m src.inference.deepsort_tracker --input 0")
    print("\n2. Start the API:")
    print("   python -m src.api.production_api")
    print("\n3. Read QUICK_START.md for more examples")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()


