"""Run real benchmark on actual MOT20 data and collect results"""
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import subprocess

# Ensure UTF-8 output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_tracking(sequence_path, output_dir):
    """Run tracking on a sequence"""
    print(f"\n▶ Processing: {sequence_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq_name = Path(sequence_path).parent.name
    
    output_video = output_dir / f"{seq_name}_tracked_{timestamp}.mp4"
    output_json = output_dir / f"{seq_name}_results_{timestamp}.json"
    
    # Run tracking
    cmd = [
        sys.executable, "-m", "src.inference.deepsort_tracker",
        "--input", str(sequence_path),
        "--output", str(output_video),
        "--detection-model", "models/checkpoints/yolov8n.pt"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"✓ Tracking completed")
            if output_video.exists():
                size_mb = output_video.stat().st_size / (1024 * 1024)
                print(f"  Video: {output_video.name} ({size_mb:.2f} MB)")
            return {
                'success': True,
                'video': str(output_video),
                'json': str(output_json),
                'sequence': seq_name
            }
        else:
            print(f"✗ Tracking failed")
            print(f"  Error: {result.stderr[:500]}")
            return {'success': False, 'error': result.stderr[:500]}
    except Exception as e:
        print(f"✗ Exception: {e}")
        return {'success': False, 'error': str(e)}

def run_evaluation(gt_path, results_path, output_dir):
    """Run MOT evaluation"""
    print(f"\n▶ Evaluating: {gt_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq_name = Path(gt_path).parent.name
    
    output_summary = output_dir / f"{seq_name}_metrics_{timestamp}.json"
    
    # Check if evaluation script exists
    eval_script = Path("scripts/evaluate_benchmark.py")
    if not eval_script.exists():
        print("⚠ Evaluation script not found, skipping...")
        return None
    
    cmd = [
        sys.executable, str(eval_script),
        "--gt-path", str(gt_path),
        "--results-path", str(results_path),
        "--output", str(output_summary)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"✓ Evaluation completed")
            if output_summary.exists():
                with open(output_summary, 'r') as f:
                    metrics = json.load(f)
                print(f"  MOTA: {metrics.get('mota', 'N/A'):.3f}")
                print(f"  MOTP: {metrics.get('motp', 'N/A'):.3f}")
                print(f"  IDF1: {metrics.get('idf1', 'N/A'):.3f}")
                return metrics
        else:
            print(f"✗ Evaluation failed: {result.stderr[:500]}")
            return None
    except Exception as e:
        print(f"✗ Exception: {e}")
        return None

def main():
    """Run complete benchmark"""
    print_header("Real Benchmark - People Tracking System")
    
    # Create output directory
    output_dir = Path("data/processed/real_benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"benchmark_report_{timestamp}.md"
    
    # Find MOT20 sequences
    mot_train = Path("data/raw/MOT20/MOT20/train")
    sequences = []
    
    if not mot_train.exists():
        print(f"✗ MOT20 train directory not found: {mot_train}")
        print("Please ensure MOT20 dataset is downloaded.")
        return
    
    # Find available sequences
    for seq_dir in sorted(mot_train.iterdir()):
        img_dir = seq_dir / "img1"
        gt_file = seq_dir / "gt" / "gt.txt"
        
        if img_dir.exists() and gt_file.exists():
            img_count = len(list(img_dir.glob("*.jpg")))
            sequences.append({
                'name': seq_dir.name,
                'img_path': str(img_dir),
                'gt_path': str(seq_dir),
                'frames': img_count
            })
    
    if not sequences:
        print("✗ No MOT20 sequences found!")
        return
    
    print(f"\nFound {len(sequences)} sequences:")
    for seq in sequences:
        print(f"  - {seq['name']}: {seq['frames']} frames")
    
    # Process sequences (start with smallest for quick test)
    sequences.sort(key=lambda x: x['frames'])
    
    print(f"\n{'='*70}")
    print("Starting benchmark...")
    print(f"{'='*70}")
    
    all_results = []
    
    # Process first sequence (smallest) for quick demo
    selected = sequences[0]
    print(f"\n🎯 Processing: {selected['name']} ({selected['frames']} frames)")
    
    # Run tracking
    track_result = run_tracking(selected['img_path'], output_dir)
    
    if track_result and track_result['success']:
        all_results.append({
            'sequence': selected['name'],
            'frames': selected['frames'],
            'tracking': track_result,
            'timestamp': timestamp
        })
        
        # Try to run evaluation if we have ground truth
        metrics = run_evaluation(
            selected['gt_path'],
            track_result.get('json', ''),
            output_dir
        )
        
        if metrics:
            all_results[-1]['metrics'] = metrics
    
    # Generate report
    print_header("Generating Report")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Real Benchmark Results - People Tracking System\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Sequences Processed:** {len(all_results)}\n")
        f.write(f"- **Total Frames:** {sum(r['frames'] for r in all_results)}\n\n")
        
        f.write("## Results\n\n")
        for result in all_results:
            f.write(f"### {result['sequence']}\n\n")
            f.write(f"- **Frames:** {result['frames']}\n")
            
            if 'tracking' in result and result['tracking'].get('success'):
                f.write(f"- **Status:** ✓ Success\n")
                f.write(f"- **Output Video:** `{Path(result['tracking']['video']).name}`\n")
            else:
                f.write(f"- **Status:** ✗ Failed\n")
            
            if 'metrics' in result:
                metrics = result['metrics']
                f.write(f"\n**Metrics:**\n")
                f.write(f"- MOTA: {metrics.get('mota', 'N/A'):.3f}\n")
                f.write(f"- MOTP: {metrics.get('motp', 'N/A'):.3f}\n")
                f.write(f"- IDF1: {metrics.get('idf1', 'N/A'):.3f}\n")
                f.write(f"- Precision: {metrics.get('precision', 'N/A'):.3f}\n")
                f.write(f"- Recall: {metrics.get('recall', 'N/A'):.3f}\n")
                f.write(f"- ID Switches: {metrics.get('id_switches', 'N/A')}\n")
            
            f.write("\n")
        
        f.write("## Files Generated\n\n")
        f.write("All output files are in: `data/processed/real_benchmark_results/`\n\n")
        
        for file in sorted(output_dir.glob("*")):
            if file.is_file() and file.suffix in ['.mp4', '.json', '.md']:
                size_mb = file.stat().st_size / (1024 * 1024)
                f.write(f"- `{file.name}` ({size_mb:.2f} MB)\n")
    
    print(f"\n✓ Report saved: {report_file}")
    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"{'='*70}")
    print(f"\nResults location: {output_dir}")
    print(f"Report: {report_file.name}")
    
    if all_results:
        print(f"\n✅ Successfully processed {len(all_results)} sequence(s)")
        for result in all_results:
            if 'tracking' in result and result['tracking'].get('success'):
                print(f"  - {result['sequence']}: ✓")

if __name__ == "__main__":
    main()

