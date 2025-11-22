"""Compare results after critical fixes"""
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
    """Compare fixed results"""
    print("="*80)
    print("🎯 CRITICAL FIXES APPLIED - Results Comparison")
    print("="*80)
    
    # Load metrics
    original = load_metrics("data/processed/real_benchmark_results/MOT20-01_metrics.json")
    improved = load_metrics("data/processed/real_benchmark_results/MOT20-01_improved_metrics.json")
    fixed = load_metrics("data/processed/real_benchmark_results/MOT20-01_fixed_metrics.json")
    
    if not fixed:
        print("\n⚠ Fixed metrics not found. Run tracking first.")
        return
    
    print("\n📊 Performance Comparison:")
    print("="*80)
    print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Fixed':<15} {'Change':<15} {'Status':<10}")
    print("-"*80)
    
    metrics = [
        ('Recall', 'recall', '%', 'higher'),
        ('MOTA', 'mota', '%', 'higher'),
        ('IDF1', 'idf1', '%', 'higher'),
        ('Precision', 'precision', '%', 'higher'),
        ('ID Switches', 'id_switches', '', 'lower'),
    ]
    
    for name, key, unit, better in metrics:
        orig_val = original.get(key, 0) if original else 0
        imp_val = improved.get(key, 0) if improved else 0
        fix_val = fixed.get(key, 0)
        
        if unit == '%':
            orig_str = f"{orig_val*100:.2f}%" if original else "N/A"
            imp_str = f"{imp_val*100:.2f}%" if improved else "N/A"
            fix_str = f"{fix_val*100:.2f}%"
            
            if original:
                change = fix_val - orig_val
                change_pct = (change / orig_val * 100) if orig_val > 0 else 0
                change_str = f"{change*100:+.2f}% ({change_pct:+.1f}%)"
            else:
                change_str = "N/A"
        else:
            orig_str = f"{orig_val:.0f}" if original else "N/A"
            imp_str = f"{imp_val:.0f}" if improved else "N/A"
            fix_str = f"{fix_val:.0f}"
            
            if original:
                change = fix_val - orig_val
                change_pct = (change / orig_val * 100) if orig_val > 0 else 0
                if key == 'id_switches':
                    change_str = f"{change:.0f} ({change_pct:+.1f}%)"
                else:
                    change_str = f"{change:+.0f} ({change_pct:+.1f}%)"
            else:
                change_str = "N/A"
        
        # Determine status
        if original:
            if better == 'lower':
                status = "✅ Better" if fix_val < orig_val else "⚠️ Worse"
            else:
                status = "✅ Better" if fix_val > orig_val else "⚠️ Worse"
        else:
            status = "N/A"
        
        print(f"{name:<20} {orig_str:<15} {imp_str:<15} {fix_str:<15} {change_str:<15} {status:<10}")
    
    # Key improvements
    print("\n" + "="*80)
    print("🎯 Key Improvements (Fixed vs Original):")
    print("="*80)
    
    if original:
        recall_orig = original.get('recall', 0)
        recall_fix = fixed.get('recall', 0)
        mota_orig = original.get('mota', 0)
        mota_fix = fixed.get('mota', 0)
        idf1_orig = original.get('idf1', 0)
        idf1_fix = fixed.get('idf1', 0)
        id_switches_orig = original.get('id_switches', 0)
        id_switches_fix = fixed.get('id_switches', 0)
        
        recall_improvement = ((recall_fix - recall_orig) / recall_orig * 100) if recall_orig > 0 else 0
        mota_improvement = ((mota_fix - mota_orig) / mota_orig * 100) if mota_orig > 0 else 0
        idf1_improvement = ((idf1_fix - idf1_orig) / idf1_orig * 100) if idf1_orig > 0 else 0
        id_switches_reduction = ((id_switches_orig - id_switches_fix) / id_switches_orig * 100) if id_switches_orig > 0 else 0
        
        print(f"\n✅ Recall: {recall_orig*100:.2f}% → {recall_fix*100:.2f}% (提升 {recall_improvement:.1f}%)")
        print(f"✅ MOTA: {mota_orig*100:.2f}% → {mota_fix*100:.2f}% (提升 {mota_improvement:.1f}%)")
        print(f"✅ IDF1: {idf1_orig*100:.2f}% → {idf1_fix*100:.2f}% (提升 {idf1_improvement:.1f}%)")
        print(f"✅ ID Switches: {id_switches_orig} → {id_switches_fix} (减少 {id_switches_reduction:.1f}%)")
    
    # Compare with improved
    if improved:
        print("\n" + "="*80)
        print("🎯 Fixed vs Improved (After Critical Fixes):")
        print("="*80)
        
        recall_imp = improved.get('recall', 0)
        recall_fix = fixed.get('recall', 0)
        mota_imp = improved.get('mota', 0)
        mota_fix = fixed.get('mota', 0)
        idf1_imp = improved.get('idf1', 0)
        idf1_fix = fixed.get('idf1', 0)
        
        if recall_fix > recall_imp:
            improvement = ((recall_fix - recall_imp) / recall_imp * 100) if recall_imp > 0 else 0
            print(f"\n✅ Recall improved: {recall_imp*100:.2f}% → {recall_fix*100:.2f}% (额外提升 {improvement:.1f}%)")
        elif recall_fix < recall_imp:
            decrease = ((recall_imp - recall_fix) / recall_imp * 100) if recall_imp > 0 else 0
            print(f"⚠️ Recall decreased: {recall_imp*100:.2f}% → {recall_fix*100:.2f}% (下降 {decrease:.1f}%)")
        
        if mota_fix > mota_imp:
            improvement = ((mota_fix - mota_imp) / mota_imp * 100) if mota_imp > 0 else 0
            print(f"✅ MOTA improved: {mota_imp*100:.2f}% → {mota_fix*100:.2f}% (额外提升 {improvement:.1f}%)")
        elif mota_fix < mota_imp:
            decrease = ((mota_imp - mota_fix) / mota_imp * 100) if mota_imp > 0 else 0
            print(f"⚠️ MOTA decreased: {mota_imp*100:.2f}% → {mota_fix*100:.2f}% (下降 {decrease:.1f}%)")
        
        if idf1_fix > idf1_imp:
            improvement = ((idf1_fix - idf1_imp) / idf1_imp * 100) if idf1_imp > 0 else 0
            print(f"✅ IDF1 improved: {idf1_imp*100:.2f}% → {idf1_fix*100:.2f}% (额外提升 {improvement:.1f}%)")
        elif idf1_fix < idf1_imp:
            decrease = ((idf1_imp - idf1_fix) / idf1_imp * 100) if idf1_imp > 0 else 0
            print(f"⚠️ IDF1 decreased: {idf1_imp*100:.2f}% → {idf1_fix*100:.2f}% (下降 {decrease:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ CRITICAL FIXES COMPLETE!")
    print("="*80)
    print("\n📂 Fixed Results:")
    print("   - Video: data/processed/real_benchmark_results/MOT20-01_fixed_20251122_112147.mp4")
    print("   - Data:  data/processed/real_benchmark_results/MOT20-01_fixed_20251122_112147.json")
    print("   - Metrics: data/processed/real_benchmark_results/MOT20-01_fixed_metrics.json")
    print("\n🎯 Critical fixes applied:")
    print("   1. Confidence filter: 0.3 → 0.15 (match detection threshold)")
    print("   2. IoU threshold: 0.3 → 0.25 (looser for crowded scenes)")
    print("   3. max_age: 40 → 50 (better track persistence)")
    print("="*80)

if __name__ == "__main__":
    main()

