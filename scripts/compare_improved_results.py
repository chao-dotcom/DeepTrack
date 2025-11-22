"""Compare improved results with previous versions"""
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
    """Compare improved results"""
    print("="*80)
    print("🎯 IMPROVED SYSTEM RESULTS - All Optimizations Applied")
    print("="*80)
    
    # Load metrics
    original = load_metrics("data/processed/real_benchmark_results/MOT20-01_metrics.json")
    final = load_metrics("data/processed/real_benchmark_results/MOT20-01_final_metrics.json")
    improved = load_metrics("data/processed/real_benchmark_results/MOT20-01_improved_metrics.json")
    
    if not improved:
        print("\n⚠ Improved metrics not found. Run tracking first.")
        return
    
    print("\n📊 Performance Comparison:")
    print("="*80)
    print(f"{'Metric':<20} {'Original':<15} {'Final':<15} {'Improved':<15} {'Change':<15} {'Status':<10}")
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
        fin_val = final.get(key, 0) if final else 0
        imp_val = improved.get(key, 0)
        
        if unit == '%':
            orig_str = f"{orig_val*100:.2f}%" if original else "N/A"
            fin_str = f"{fin_val*100:.2f}%" if final else "N/A"
            imp_str = f"{imp_val*100:.2f}%"
            
            if original:
                change = imp_val - orig_val
                change_pct = (change / orig_val * 100) if orig_val > 0 else 0
                change_str = f"{change*100:+.2f}% ({change_pct:+.1f}%)"
            else:
                change_str = "N/A"
        else:
            orig_str = f"{orig_val:.0f}" if original else "N/A"
            fin_str = f"{fin_val:.0f}" if final else "N/A"
            imp_str = f"{imp_val:.0f}"
            
            if original:
                change = imp_val - orig_val
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
                status = "✅ Better" if imp_val < orig_val else "⚠️ Worse"
            else:
                status = "✅ Better" if imp_val > orig_val else "⚠️ Worse"
        else:
            status = "N/A"
        
        print(f"{name:<20} {orig_str:<15} {fin_str:<15} {imp_str:<15} {change_str:<15} {status:<10}")
    
    # Key improvements
    print("\n" + "="*80)
    print("🎯 Key Improvements (Improved vs Original):")
    print("="*80)
    
    if original:
        recall_orig = original.get('recall', 0)
        recall_imp = improved.get('recall', 0)
        mota_orig = original.get('mota', 0)
        mota_imp = improved.get('mota', 0)
        idf1_orig = original.get('idf1', 0)
        idf1_imp = improved.get('idf1', 0)
        id_switches_orig = original.get('id_switches', 0)
        id_switches_imp = improved.get('id_switches', 0)
        
        recall_improvement = ((recall_imp - recall_orig) / recall_orig * 100) if recall_orig > 0 else 0
        mota_improvement = ((mota_imp - mota_orig) / mota_orig * 100) if mota_orig > 0 else 0
        idf1_improvement = ((idf1_imp - idf1_orig) / idf1_orig * 100) if idf1_orig > 0 else 0
        id_switches_reduction = ((id_switches_orig - id_switches_imp) / id_switches_orig * 100) if id_switches_orig > 0 else 0
        
        print(f"\n✅ Recall: {recall_orig*100:.2f}% → {recall_imp*100:.2f}% (提升 {recall_improvement:.1f}%)")
        print(f"✅ MOTA: {mota_orig*100:.2f}% → {mota_imp*100:.2f}% (提升 {mota_improvement:.1f}%)")
        print(f"✅ IDF1: {idf1_orig*100:.2f}% → {idf1_imp*100:.2f}% (提升 {idf1_improvement:.1f}%)")
        print(f"✅ ID Switches: {id_switches_orig} → {id_switches_imp} (减少 {id_switches_reduction:.1f}%)")
    
    # Compare with final
    if final:
        print("\n" + "="*80)
        print("🎯 Improved vs Final (Previous Best):")
        print("="*80)
        
        recall_fin = final.get('recall', 0)
        recall_imp = improved.get('recall', 0)
        mota_fin = final.get('mota', 0)
        mota_imp = improved.get('mota', 0)
        idf1_fin = final.get('idf1', 0)
        idf1_imp = improved.get('idf1', 0)
        
        if recall_imp > recall_fin:
            improvement = ((recall_imp - recall_fin) / recall_fin * 100) if recall_fin > 0 else 0
            print(f"\n✅ Recall further improved: {recall_fin*100:.2f}% → {recall_imp*100:.2f}% (额外提升 {improvement:.1f}%)")
        
        if mota_imp > mota_fin:
            improvement = ((mota_imp - mota_fin) / mota_fin * 100) if mota_fin > 0 else 0
            print(f"✅ MOTA further improved: {mota_fin*100:.2f}% → {mota_imp*100:.2f}% (额外提升 {improvement:.1f}%)")
        
        if idf1_imp > idf1_fin:
            improvement = ((idf1_imp - idf1_fin) / idf1_fin * 100) if idf1_fin > 0 else 0
            print(f"✅ IDF1 further improved: {idf1_fin*100:.2f}% → {idf1_imp*100:.2f}% (额外提升 {improvement:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\n📂 Improved Results:")
    print("   - Video: data/processed/real_benchmark_results/MOT20-01_improved_20251122_111322.mp4")
    print("   - Data:  data/processed/real_benchmark_results/MOT20-01_improved_20251122_111322.json")
    print("   - Metrics: data/processed/real_benchmark_results/MOT20-01_improved_metrics.json")
    print("\n🎯 All optimizations from docs/6.md have been applied!")
    print("="*80)

if __name__ == "__main__":
    main()

