"""Complete test script to generate results and outputs"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n▶ {description}")
    print(f"  Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False

def main():
    """Run complete test suite"""
    print_header("People Tracking System - Complete Test Suite")
    
    # Create output directory
    output_dir = Path("data/processed/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_report = output_dir / f"test_report_{timestamp}.txt"
    
    with open(test_report, 'w', encoding='utf-8') as report:
        report.write("People Tracking System - Test Report\n")
        report.write(f"Generated: {datetime.now().isoformat()}\n")
        report.write("="*70 + "\n\n")
        
        # Test 1: Check environment
        print_header("Test 1: Environment Check")
        report.write("Test 1: Environment Check\n")
        report.write("-"*70 + "\n")
        
        checks = [
            ("Python version", "python --version"),
            ("Virtual environment", "python -c \"import sys; print(sys.prefix)\""),
            ("Required packages", "python -c \"import cv2, fastapi, streamlit, ultralytics; print('All packages OK')\""),
        ]
        
        for name, cmd in checks:
            print(f"\nChecking {name}...")
            result = run_command(cmd, f"Check {name}")
            report.write(f"{name}: {'✓ PASS' if result else '✗ FAIL'}\n")
        
        # Test 2: Model availability
        print_header("Test 2: Model Check")
        report.write("\nTest 2: Model Check\n")
        report.write("-"*70 + "\n")
        
        model_path = Path("models/checkpoints/yolov8n.pt")
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model found: {model_path} ({size_mb:.2f} MB)")
            report.write(f"Model: ✓ Found ({size_mb:.2f} MB)\n")
        else:
            print("✗ Model not found. Downloading...")
            result = run_command("python scripts/download_models.py", "Download YOLOv8 model")
            report.write(f"Model download: {'✓ PASS' if result else '✗ FAIL'}\n")
        
        # Test 3: MOT20 dataset check
        print_header("Test 3: MOT20 Dataset Check")
        report.write("\nTest 3: MOT20 Dataset Check\n")
        report.write("-"*70 + "\n")
        
        mot_path = Path("data/raw/MOT20/MOT20")
        if mot_path.exists():
            sequences = list((mot_path / "train").glob("MOT20-*"))
            sequences += list((mot_path / "test").glob("MOT20-*"))
            print(f"✓ Found {len(sequences)} MOT20 sequences")
            report.write(f"MOT20 sequences: ✓ Found {len(sequences)}\n")
            for seq in sequences[:5]:  # List first 5
                report.write(f"  - {seq.name}\n")
        else:
            print("✗ MOT20 dataset not found")
            report.write("MOT20 dataset: ✗ NOT FOUND\n")
        
        # Test 4: Process MOT20 sequence
        print_header("Test 4: Process MOT20 Sequence")
        report.write("\nTest 4: Process MOT20 Sequence\n")
        report.write("-"*70 + "\n")
        
        test_sequence = "data/raw/MOT20/MOT20/train/MOT20-01/img1"
        if Path(test_sequence).exists():
            output_video = output_dir / f"MOT20-01_tracked_{timestamp}.mp4"
            output_json = output_dir / f"MOT20-01_results_{timestamp}.json"
            
            cmd = f'python -m src.inference.main --input "{test_sequence}" --output "{output_video}" --model models/checkpoints/yolov8n.pt'
            
            print(f"Processing {test_sequence}...")
            print("This may take a few minutes...")
            
            result = run_command(cmd, "Process MOT20-01 sequence")
            
            if result and output_video.exists():
                size_mb = output_video.stat().st_size / (1024 * 1024)
                print(f"✓ Output video created: {output_video} ({size_mb:.2f} MB)")
                report.write(f"Processing: ✓ PASS\n")
                report.write(f"Output video: {output_video} ({size_mb:.2f} MB)\n")
                
                if output_json.exists():
                    report.write(f"Results JSON: {output_json}\n")
            else:
                print("✗ Processing failed")
                report.write("Processing: ✗ FAIL\n")
        else:
            print(f"✗ Test sequence not found: {test_sequence}")
            report.write("Test sequence: ✗ NOT FOUND\n")
        
        # Test 5: API Server (quick test)
        print_header("Test 5: API Server Test")
        report.write("\nTest 5: API Server Test\n")
        report.write("-"*70 + "\n")
        
        print("Note: API server test requires manual verification")
        print("Start API server with: python -m src.api.main")
        print("Then visit: http://127.0.0.1:8000/docs")
        report.write("API Server: Manual test required\n")
        report.write("  Start: python -m src.api.main\n")
        report.write("  URL: http://127.0.0.1:8000/docs\n")
        
        # Summary
        print_header("Test Summary")
        report.write("\n" + "="*70 + "\n")
        report.write("Test Summary\n")
        report.write("="*70 + "\n")
        report.write(f"Test completed: {datetime.now().isoformat()}\n")
        report.write(f"Results directory: {output_dir}\n")
        report.write("\nGenerated files:\n")
        
        for file in output_dir.glob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                report.write(f"  - {file.name} ({size_mb:.2f} MB)\n")
    
    print(f"\n✓ Test report saved to: {test_report}")
    print(f"\nAll output files are in: {output_dir}")
    print("\nYou can now show these results:")
    print(f"  1. Test report: {test_report}")
    print(f"  2. Output video: {output_dir / f'MOT20-01_tracked_{timestamp}.mp4'}")
    print(f"  3. Results JSON: {output_dir / f'MOT20-01_results_{timestamp}.json'}")

if __name__ == "__main__":
    main()

