"""
Generate all results for docs/5.md
Runs benchmark, collects results, creates visualizations
"""
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


def main():
    """Main function to generate all results"""
    print("="*70)
    print("Generating All Results for docs/5.md")
    print("="*70)
    
    results = {
        'benchmark': False,
        'results_collection': False,
        'comparison_table': False,
        'graphs': False,
        'report': False
    }
    
    # Step 1: Run benchmark
    print("\n[1/5] Running MOT20 Benchmark...")
    print("-" * 70)
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_full_benchmark.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            results['benchmark'] = True
            print("[OK] Benchmark completed")
        else:
            print(f"[WARNING] Benchmark had issues: {result.stderr[:200]}")
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
    
    # Step 2: Collect results
    print("\n[2/5] Collecting Results...")
    print("-" * 70)
    summary_path = Path("outputs/mot_results/summary.json")
    if summary_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, "scripts/collect_benchmark_results.py",
                 "--summary", str(summary_path),
                 "--output-dir", "results"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                results['results_collection'] = True
                print("[OK] Results collected")
            else:
                print(f"[WARNING] Collection had issues")
        except Exception as e:
            print(f"[ERROR] Collection failed: {e}")
    else:
        print("[WARNING] Summary file not found. Run benchmark first.")
    
    # Step 3: Check generated files
    print("\n[3/5] Verifying Generated Files...")
    print("-" * 70)
    
    results_dir = Path("results")
    if results_dir.exists():
        csv_files = list(results_dir.glob("comparison_table_*.csv"))
        md_files = list(results_dir.glob("comparison_table_*.md"))
        graphs_dir = results_dir / "graphs"
        
        if csv_files:
            results['comparison_table'] = True
            print(f"[OK] Found {len(csv_files)} comparison table(s)")
        if md_files:
            print(f"[OK] Found {len(md_files)} markdown table(s)")
        if graphs_dir.exists() and list(graphs_dir.glob("*.png")):
            results['graphs'] = True
            print(f"[OK] Found graphs in {graphs_dir}")
        
        report_files = list(results_dir.glob("benchmark_report_*.md"))
        if report_files:
            results['report'] = True
            print(f"[OK] Found {len(report_files)} report(s)")
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    total = len(results)
    completed = sum(results.values())
    
    for task, status in results.items():
        status_icon = "[OK]" if status else "[FAIL]"
        print(f"{status_icon} {task.replace('_', ' ').title()}")
    
    print(f"\nCompleted: {completed}/{total} tasks")
    
    if completed == total:
        print("\n[SUCCESS] All results generated successfully!")
        print("\nGenerated files:")
        print("  - results/comparison_table_*.csv")
        print("  - results/comparison_table_*.md")
        print("  - results/graphs/*.png")
        print("  - results/benchmark_report_*.md")
    else:
        print(f"\n[WARNING] Some tasks incomplete. Check errors above.")


if __name__ == '__main__':
    main()

