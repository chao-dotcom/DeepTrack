"""
Run full benchmark evaluation and collect all results
Implements Priority 1 from docs/5.md
"""
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


def run_benchmark():
    """Run the benchmark evaluation"""
    print("="*60)
    print("Running MOT20 Benchmark Evaluation")
    print("="*60)
    
    # Check if MOT20 data exists
    mot20_path = Path("data/raw/MOT20/MOT20/train")
    if not mot20_path.exists():
        print(f"⚠️  MOT20 data not found at: {mot20_path}")
        print("Please download MOT20 dataset first.")
        return False
    
    # Run benchmark
    cmd = [
        sys.executable,
        "scripts/evaluate_benchmark.py",
        "--gt-path", str(mot20_path),
        "--output-path", "outputs/mot_results",
        "--detection-model", "models/checkpoints/yolov8n.pt"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Benchmark completed successfully!")
        return True
    else:
        print("\n❌ Benchmark failed!")
        return False


def collect_results():
    """Collect and organize results"""
    print("\n" + "="*60)
    print("Collecting and Organizing Results")
    print("="*60)
    
    summary_path = Path("outputs/mot_results/summary.json")
    if not summary_path.exists():
        print(f"⚠️  Summary file not found: {summary_path}")
        return False
    
    # Run results collector
    cmd = [
        sys.executable,
        "scripts/collect_benchmark_results.py",
        "--summary", str(summary_path),
        "--output-dir", "results"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Results collected successfully!")
        return True
    else:
        print("\n❌ Results collection failed!")
        return False


def main():
    """Main function"""
    print("\n" + "="*60)
    print("Full Benchmark Pipeline - docs/5.md Priority 1")
    print("="*60)
    
    # Step 1: Run benchmark
    if not run_benchmark():
        print("\n❌ Benchmark failed. Exiting.")
        return
    
    # Step 2: Collect results
    if not collect_results():
        print("\n❌ Results collection failed. Exiting.")
        return
    
    print("\n" + "="*60)
    print("✅ Complete! Results available in 'results/' directory")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/comparison_table_*.csv")
    print("  - results/comparison_table_*.md")
    print("  - results/graphs/*.png")
    print("  - results/benchmark_report_*.md")


if __name__ == '__main__':
    main()

