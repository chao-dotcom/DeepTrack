"""Compare all versions: original, fixed, and enhanced"""
import json
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_metrics(filepath):
    """Load metrics from JSON"""
    if not Path(filepath).exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def load_stats(json_path):
    """Load tracking statistics"""
    if not Path(json_path).exists():
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('statistics', {})

def main():
    """Compare all three versions"""
    print("="*70)
    print("Complete Comparison: Original vs Fixed vs Enhanced")
    print("="*70)
    
    # Load all metrics
    original = load_metrics("data/processed/real_benchmark_results/MOT20-01_metrics.json")
    fixed = load_metrics("data/processed/real_benchmark_results/MOT20-01_fixed_metrics.json")
    enhanced = load_metrics("data/processed/real_benchmark_results/MOT20-01_enhanced_metrics.json")
    
    # Load all stats
    original_stats = load_stats("data/processed/real_benchmark_results/MOT20-01_tracked_20251122_100146.json")
    fixed_stats = load_stats("data/processed/real_benchmark_results/MOT20-01_fixed_20251122_102837.json")
    enhanced_stats = load_stats("data/processed/real_benchmark_results/MOT20-01_enhanced_20251122_103730.json")
    
    if not all([original, fixed, enhanced]):
        print("\n⚠ Some metrics files missing. Run all versions first.")
        return
    
    print("\n📊 Performance Metrics Comparison:")
    print("-"*70)
    print(f"{'Metric':<20} {'Original':<15} {'Fixed':<15} {'Enhanced':<15} {'Best':<10}")
    print("-"*70)
    
    metrics = [
        ('ID Switches', 'id_switches', '', 'lower'),
        ('IDF1', 'idf1', '%', 'higher'),
        ('MOTA', 'mota', '%', 'higher'),
        ('MOTP', 'motp', '%', 'higher'),
        ('Precision', 'precision', '%', 'higher'),
        ('Recall', 'recall', '%', 'higher'),
    ]
    
    for name, key, unit, better in metrics:
        orig_val = original.get(key, 0)
        fixed_val = fixed.get(key, 0)
        enh_val = enhanced.get(key, 0)
        
        if unit == '%':
            orig_str = f"{orig_val*100:.2f}%"
            fixed_str = f"{fixed_val*100:.2f}%"
            enh_str = f"{enh_val*100:.2f}%"
        else:
            orig_str = f"{orig_val:.0f}"
            fixed_str = f"{fixed_val:.0f}"
            enh_str = f"{enh_val:.0f}"
        
        # Determine best
        if better == 'lower':
            best_val = min(orig_val, fixed_val, enh_val)
            if best_val == orig_val:
                best = "Original"
            elif best_val == fixed_val:
                best = "Fixed"
            else:
                best = "Enhanced"
        else:
            best_val = max(orig_val, fixed_val, enh_val)
            if best_val == orig_val:
                best = "Original"
            elif best_val == fixed_val:
                best = "Fixed"
            else:
                best = "Enhanced"
        
        print(f"{name:<20} {orig_str:<15} {fixed_str:<15} {enh_str:<15} {best:<10}")
    
    # Statistics comparison
    if all([original_stats, fixed_stats, enhanced_stats]):
        print("\n" + "-"*70)
        print("📈 Tracking Statistics Comparison:")
        print("-"*70)
        print(f"{'Metric':<25} {'Original':<15} {'Fixed':<15} {'Enhanced':<15}")
        print("-"*70)
        
        stats_metrics = [
            ('Total Tracks', 'total_tracks'),
            ('Avg Track Length', 'avg_track_length'),
            ('Max Track Length', 'max_track_length'),
        ]
        
        for name, key in stats_metrics:
            orig = original_stats.get(key, 0)
            fix = fixed_stats.get(key, 0)
            enh = enhanced_stats.get(key, 0)
            
            if key == 'avg_track_length':
                orig_str = f"{orig:.2f} frames"
                fix_str = f"{fix:.2f} frames"
                enh_str = f"{enh:.2f} frames"
            else:
                orig_str = f"{orig:.0f}"
                fix_str = f"{fix:.0f}"
                enh_str = f"{enh:.0f}"
            
            print(f"{name:<25} {orig_str:<15} {fix_str:<15} {enh_str:<15}")
    
    # Key improvements
    print("\n" + "="*70)
    print("🎯 Key Improvements (Enhanced vs Original):")
    print("="*70)
    
    id_switches_orig = original.get('id_switches', 0)
    id_switches_enh = enhanced.get('id_switches', 0)
    idf1_orig = original.get('idf1', 0)
    idf1_enh = enhanced.get('idf1', 0)
    
    if id_switches_enh < id_switches_orig:
        reduction = ((id_switches_orig - id_switches_enh) / id_switches_orig) * 100
        print(f"\n✅ ID Switches: {id_switches_orig} → {id_switches_enh} (减少 {reduction:.1f}%)")
    
    if idf1_enh > idf1_orig:
        improvement = ((idf1_enh - idf1_orig) / idf1_orig) * 100
        print(f"✅ IDF1: {idf1_orig*100:.2f}% → {idf1_enh*100:.2f}% (提升 {improvement:.1f}%)")
    
    # Compare with fixed version
    print("\n" + "="*70)
    print("🎯 Enhanced vs Fixed Version:")
    print("="*70)
    
    id_switches_fixed = fixed.get('id_switches', 0)
    idf1_fixed = fixed.get('idf1', 0)
    
    if id_switches_enh < id_switches_fixed:
        reduction = ((id_switches_fixed - id_switches_enh) / id_switches_fixed) * 100
        print(f"\n✅ ID Switches further reduced: {id_switches_fixed} → {id_switches_enh} (额外减少 {reduction:.1f}%)")
    elif id_switches_enh > id_switches_fixed:
        increase = ((id_switches_enh - id_switches_fixed) / id_switches_fixed) * 100
        print(f"\n⚠️ ID Switches increased: {id_switches_fixed} → {id_switches_enh} (增加 {increase:.1f}%)")
    
    if idf1_enh > idf1_fixed:
        improvement = ((idf1_enh - idf1_fixed) / idf1_fixed) * 100
        print(f"✅ IDF1 further improved: {idf1_fixed*100:.2f}% → {idf1_enh*100:.2f}% (额外提升 {improvement:.1f}%)")
    elif idf1_enh < idf1_fixed:
        decrease = ((idf1_fixed - idf1_enh) / idf1_fixed) * 100
        print(f"⚠️ IDF1 decreased: {idf1_fixed*100:.2f}% → {idf1_enh*100:.2f}% (下降 {decrease:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70)

if __name__ == "__main__":
    main()

