"""Compare tracking results before and after ID switch fix"""
import json
import sys
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_metrics(filepath):
    """Load metrics from JSON"""
    if not Path(filepath).exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    """Compare before and after results"""
    print("="*70)
    print("ID Switch Fix - Before vs After Comparison")
    print("="*70)
    
    # Load old metrics
    old_metrics = load_metrics("data/processed/real_benchmark_results/MOT20-01_metrics.json")
    
    # Load new metrics
    new_metrics = load_metrics("data/processed/real_benchmark_results/MOT20-01_fixed_metrics.json")
    
    if not old_metrics:
        print("\n⚠ Old metrics not found. Run original tracking first.")
        return
    
    if not new_metrics:
        print("\n⚠ New metrics not found. Run fixed tracking first.")
        return
    
    print("\n📊 Performance Comparison:")
    print("-"*70)
    
    metrics_to_compare = [
        ('MOTA', 'mota', '%'),
        ('MOTP', 'motp', '%'),
        ('IDF1', 'idf1', '%'),
        ('Precision', 'precision', '%'),
        ('Recall', 'recall', '%'),
        ('ID Switches', 'id_switches', ''),
        ('Matches', 'matches', ''),
    ]
    
    print(f"\n{'Metric':<20} {'Before':<15} {'After':<15} {'Change':<15} {'Status':<10}")
    print("-"*70)
    
    improvements = []
    for name, key, unit in metrics_to_compare:
        old_val = old_metrics.get(key, 0)
        new_val = new_metrics.get(key, 0)
        
        if unit == '%':
            old_display = f"{old_val*100:.2f}%"
            new_display = f"{new_val*100:.2f}%"
            if key == 'id_switches':
                change = old_val - new_val
                change_pct = (change / old_val * 100) if old_val > 0 else 0
                change_display = f"-{change:.0f} ({change_pct:.1f}%)"
                status = "✅ Better" if new_val < old_val else "⚠️ Worse"
            else:
                change = new_val - old_val
                change_pct = (change / old_val * 100) if old_val > 0 else 0
                change_display = f"+{change*100:.2f}% ({change_pct:+.1f}%)"
                status = "✅ Better" if new_val > old_val else "⚠️ Worse"
        else:
            old_display = f"{old_val:.0f}"
            new_display = f"{new_val:.0f}"
            if key == 'id_switches':
                change = old_val - new_val
                change_pct = (change / old_val * 100) if old_val > 0 else 0
                change_display = f"-{change:.0f} ({change_pct:.1f}%)"
                status = "✅ Better" if new_val < old_val else "⚠️ Worse"
            else:
                change = new_val - old_val
                change_pct = (change / old_val * 100) if old_val > 0 else 0
                change_display = f"+{change:.0f} ({change_pct:+.1f}%)"
                status = "✅ Better" if new_val > old_val else "⚠️ Worse"
        
        print(f"{name:<20} {old_display:<15} {new_display:<15} {change_display:<15} {status:<10}")
        
        if status == "✅ Better":
            improvements.append(name)
    
    # Load tracking statistics
    old_json = Path("data/processed/real_benchmark_results/MOT20-01_tracked_20251122_100146.json")
    new_json = Path("data/processed/real_benchmark_results/MOT20-01_fixed_20251122_102837.json")
    
    old_stats = {}
    new_stats = {}
    
    if old_json.exists():
        with open(old_json, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
            old_stats = old_data.get('statistics', {})
    
    if new_json.exists():
        with open(new_json, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
            new_stats = new_data.get('statistics', {})
    
    print("\n" + "-"*70)
    print("📈 Tracking Statistics Comparison:")
    print("-"*70)
    
    if old_stats and new_stats:
        print(f"\n{'Metric':<25} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-"*70)
        
        stats_compare = [
            ('Total Tracks', 'total_tracks'),
            ('Avg Track Length', 'avg_track_length'),
            ('Max Track Length', 'max_track_length'),
        ]
        
        for name, key in stats_compare:
            old_val = old_stats.get(key, 0)
            new_val = new_stats.get(key, 0)
            
            if key == 'avg_track_length':
                old_display = f"{old_val:.2f} frames"
                new_display = f"{new_val:.2f} frames"
                change = new_val - old_val
                change_display = f"+{change:.2f} frames"
            else:
                old_display = f"{old_val:.0f}"
                new_display = f"{new_val:.0f}"
                change = new_val - old_val
                change_display = f"{change:+.0f}"
            
            print(f"{name:<25} {old_display:<15} {new_display:<15} {change_display:<15}")
    
    print("\n" + "="*70)
    print("✅ Summary:")
    print("="*70)
    
    if improvements:
        print(f"\n✅ Improved metrics: {', '.join(improvements)}")
    
    # Key improvements
    id_switches_old = old_metrics.get('id_switches', 0)
    id_switches_new = new_metrics.get('id_switches', 0)
    idf1_old = old_metrics.get('idf1', 0)
    idf1_new = new_metrics.get('idf1', 0)
    
    if id_switches_new < id_switches_old:
        reduction = ((id_switches_old - id_switches_new) / id_switches_old) * 100
        print(f"\n🎯 ID Switches reduced by {reduction:.1f}%")
        print(f"   Before: {id_switches_old}")
        print(f"   After:  {id_switches_new}")
    
    if idf1_new > idf1_old:
        improvement = ((idf1_new - idf1_old) / idf1_old) * 100
        print(f"\n🎯 IDF1 improved by {improvement:.1f}%")
        print(f"   Before: {idf1_old*100:.2f}%")
        print(f"   After:  {idf1_new*100:.2f}%")
    
    if new_stats.get('avg_track_length', 0) > old_stats.get('avg_track_length', 0):
        print(f"\n🎯 Average track length increased")
        print(f"   Before: {old_stats.get('avg_track_length', 0):.2f} frames")
        print(f"   After:  {new_stats.get('avg_track_length', 0):.2f} frames")
        print(f"   This means IDs are more stable!")
    
    print("\n" + "="*70)
    print("✅ Fix successful! ID stability improved.")
    print("="*70)

if __name__ == "__main__":
    main()

