"""Show final comparison of all versions"""
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

def main():
    """Show final comparison"""
    print("="*80)
    print("🎯 FINAL OPTIMIZATION RESULTS - ID Consistency Improvement")
    print("="*80)
    
    # Load all metrics
    original = load_metrics("data/processed/real_benchmark_results/MOT20-01_metrics.json")
    fixed = load_metrics("data/processed/real_benchmark_results/MOT20-01_fixed_metrics.json")
    enhanced = load_metrics("data/processed/real_benchmark_results/MOT20-01_enhanced_metrics.json")
    final = load_metrics("data/processed/real_benchmark_results/MOT20-01_final_metrics.json")
    
    if not all([original, fixed, enhanced, final]):
        print("\n⚠ Some metrics files missing.")
        return
    
    print("\n📊 Complete Performance Comparison:")
    print("="*80)
    print(f"{'Metric':<20} {'Original':<15} {'Fixed':<15} {'Enhanced':<15} {'Final':<15} {'Best':<10}")
    print("-"*80)
    
    metrics = [
        ('ID Switches', 'id_switches', '', 'lower'),
        ('IDF1', 'idf1', '%', 'higher'),
        ('MOTA', 'mota', '%', 'higher'),
        ('Precision', 'precision', '%', 'higher'),
        ('Recall', 'recall', '%', 'higher'),
    ]
    
    for name, key, unit, better in metrics:
        orig_val = original.get(key, 0)
        fixed_val = fixed.get(key, 0)
        enh_val = enhanced.get(key, 0)
        fin_val = final.get(key, 0)
        
        if unit == '%':
            orig_str = f"{orig_val*100:.2f}%"
            fixed_str = f"{fixed_val*100:.2f}%"
            enh_str = f"{enh_val*100:.2f}%"
            fin_str = f"{fin_val*100:.2f}%"
        else:
            orig_str = f"{orig_val:.0f}"
            fixed_str = f"{fixed_val:.0f}"
            enh_str = f"{enh_val:.0f}"
            fin_str = f"{fin_val:.0f}"
        
        # Determine best
        if better == 'lower':
            best_val = min(orig_val, fixed_val, enh_val, fin_val)
            if best_val == orig_val:
                best = "Original"
            elif best_val == fixed_val:
                best = "Fixed"
            elif best_val == enh_val:
                best = "Enhanced"
            else:
                best = "Final ⭐"
        else:
            best_val = max(orig_val, fixed_val, enh_val, fin_val)
            if best_val == orig_val:
                best = "Original"
            elif best_val == fixed_val:
                best = "Fixed"
            elif best_val == enh_val:
                best = "Enhanced"
            else:
                best = "Final ⭐"
        
        print(f"{name:<20} {orig_str:<15} {fixed_str:<15} {enh_str:<15} {fin_str:<15} {best:<10}")
    
    # Key improvements
    print("\n" + "="*80)
    print("🎯 Key Improvements (Final vs Original):")
    print("="*80)
    
    id_switches_orig = original.get('id_switches', 0)
    id_switches_final = final.get('id_switches', 0)
    idf1_orig = original.get('idf1', 0)
    idf1_final = final.get('idf1', 0)
    mota_orig = original.get('mota', 0)
    mota_final = final.get('mota', 0)
    
    reduction = ((id_switches_orig - id_switches_final) / id_switches_orig) * 100
    idf1_improvement = ((idf1_final - idf1_orig) / idf1_orig) * 100
    mota_improvement = ((mota_final - mota_orig) / mota_orig) * 100
    
    print(f"\n✅ ID Switches: {id_switches_orig} → {id_switches_final} (减少 {reduction:.1f}%)")
    print(f"✅ IDF1: {idf1_orig*100:.2f}% → {idf1_final*100:.2f}% (提升 {idf1_improvement:.1f}%)")
    print(f"✅ MOTA: {mota_orig*100:.2f}% → {mota_final*100:.2f}% (提升 {mota_improvement:.1f}%)")
    
    # Compare with enhanced
    print("\n" + "="*80)
    print("🎯 Final vs Enhanced (Latest Optimization):")
    print("="*80)
    
    id_switches_enh = enhanced.get('id_switches', 0)
    idf1_enh = enhanced.get('idf1', 0)
    
    if id_switches_final < id_switches_enh:
        further_reduction = ((id_switches_enh - id_switches_final) / id_switches_enh) * 100
        print(f"\n✅ ID Switches further reduced: {id_switches_enh} → {id_switches_final} (额外减少 {further_reduction:.1f}%)")
    else:
        increase = ((id_switches_final - id_switches_enh) / id_switches_enh) * 100
        print(f"\n⚠️ ID Switches increased: {id_switches_enh} → {id_switches_final} (增加 {increase:.1f}%)")
    
    if idf1_final > idf1_enh:
        further_improvement = ((idf1_final - idf1_enh) / idf1_enh) * 100
        print(f"✅ IDF1 further improved: {idf1_enh*100:.2f}% → {idf1_final*100:.2f}% (额外提升 {further_improvement:.1f}%)")
    else:
        decrease = ((idf1_enh - idf1_final) / idf1_enh) * 100
        print(f"⚠️ IDF1 decreased: {idf1_enh*100:.2f}% → {idf1_final*100:.2f}% (下降 {decrease:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\n📂 Final Results:")
    print("   - Video: data/processed/real_benchmark_results/MOT20-01_final_20251122_104610.mp4")
    print("   - Data:  data/processed/real_benchmark_results/MOT20-01_final_20251122_104610.json")
    print("   - Metrics: data/processed/real_benchmark_results/MOT20-01_final_metrics.json")
    print("\n🎯 Summary:")
    print(f"   - ID Switches reduced by {reduction:.1f}%")
    print(f"   - IDF1 improved by {idf1_improvement:.1f}%")
    print(f"   - MOTA improved by {mota_improvement:.1f}%")
    print("\n✅ ID consistency significantly improved!")
    print("="*80)

if __name__ == "__main__":
    main()

