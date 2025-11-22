"""
Collect and organize benchmark results for docs/5.md Priority 1
Creates quantitative results, comparison tables, and performance graphs
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsCollector:
    """Collect and visualize benchmark results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_benchmark_results(self, summary_path: str) -> Dict:
        """Load benchmark results from JSON"""
        with open(summary_path, 'r') as f:
            data = json.load(f)
        return data
    
    def create_comparison_table(self, results: Dict, baseline_results: Dict = None) -> pd.DataFrame:
        """
        Create comparison table against baselines
        
        Baseline references (typical values):
        - Simple Tracker: MOTA ~0.40-0.50
        - DeepSORT: MOTA ~0.60-0.70
        - SOTA: MOTA ~0.75-0.85
        """
        sequences = []
        metrics = []
        
        # Our results
        for seq_name, seq_results in results.get('per_sequence', {}).items():
            sequences.append(seq_name)
            metrics.append({
                'Sequence': seq_name,
                'MOTA': seq_results.get('mota', 0),
                'MOTP': seq_results.get('motp', 0),
                'IDF1': seq_results.get('idf1', 0),
                'Precision': seq_results.get('precision', 0),
                'Recall': seq_results.get('recall', 0),
                'ID_Switches': seq_results.get('id_switches', 0)
            })
        
        # Add average
        if results.get('average'):
            avg = results['average']
            sequences.append('Average')
            metrics.append({
                'Sequence': 'Average',
                'MOTA': avg.get('mota', 0),
                'MOTP': avg.get('motp', 0),
                'IDF1': avg.get('idf1', 0),
                'Precision': avg.get('precision', 0),
                'Recall': avg.get('recall', 0),
                'ID_Switches': avg.get('id_switches', 0)
            })
        
        df = pd.DataFrame(metrics)
        
        # Add baseline comparison if provided
        if baseline_results:
            df['Baseline_MOTA'] = baseline_results.get('mota', 0.50)
            df['Improvement'] = df['MOTA'] - df['Baseline_MOTA']
            df['Improvement_%'] = (df['Improvement'] / df['Baseline_MOTA'] * 100).round(1)
        
        return df
    
    def save_comparison_table(self, df: pd.DataFrame, format: str = 'both'):
        """Save comparison table in multiple formats"""
        if format in ['csv', 'both']:
            csv_path = self.results_dir / f'comparison_table_{self.timestamp}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")
        
        if format in ['markdown', 'both']:
            md_path = self.results_dir / f'comparison_table_{self.timestamp}.md'
            with open(md_path, 'w') as f:
                f.write("# Benchmark Results Comparison\n\n")
                f.write(df.to_markdown(index=False))
            print(f"Saved Markdown: {md_path}")
        
        if format in ['latex', 'both']:
            latex_path = self.results_dir / f'comparison_table_{self.timestamp}.tex'
            with open(latex_path, 'w') as f:
                f.write(df.to_latex(index=False, float_format="%.3f"))
            print(f"Saved LaTeX: {latex_path}")
    
    def create_performance_graphs(self, results: Dict):
        """Create performance metric graphs"""
        fig_dir = self.results_dir / 'graphs'
        fig_dir.mkdir(exist_ok=True)
        
        # Extract data
        sequences = []
        mota_values = []
        motp_values = []
        idf1_values = []
        precision_values = []
        recall_values = []
        
        for seq_name, seq_results in results.get('per_sequence', {}).items():
            sequences.append(seq_name)
            mota_values.append(seq_results.get('mota', 0))
            motp_values.append(seq_results.get('motp', 0))
            idf1_values.append(seq_results.get('idf1', 0))
            precision_values.append(seq_results.get('precision', 0))
            recall_values.append(seq_results.get('recall', 0))
        
        # 1. MOTA comparison across sequences
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sequences))
        width = 0.6
        bars = ax.bar(x, mota_values, width, label='MOTA', color='#2ecc71')
        ax.set_xlabel('Sequence', fontsize=12)
        ax.set_ylabel('MOTA Score', fontsize=12)
        ax.set_title('MOTA Performance Across MOT20 Sequences', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'mota_comparison_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. All metrics comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(sequences))
        width = 0.15
        
        metrics_data = {
            'MOTA': mota_values,
            'MOTP': motp_values,
            'IDF1': idf1_values,
            'Precision': precision_values,
            'Recall': recall_values
        }
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            offset = (i - 2) * width
            ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        
        ax.set_xlabel('Sequence', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comprehensive Performance Metrics Across Sequences', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'all_metrics_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Radar chart for average metrics
        if results.get('average'):
            avg = results['average']
            categories = ['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall']
            values = [
                avg.get('mota', 0),
                avg.get('motp', 0),
                avg.get('idf1', 0),
                avg.get('precision', 0),
                avg.get('recall', 0)
            ]
            
            # Number of variables
            N = len(categories)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Add values
            values += values[:1]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, label='Our System', color='#2ecc71')
            ax.fill(angles, values, alpha=0.25, color='#2ecc71')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim([0, 1])
            ax.set_title('Average Performance Metrics (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'radar_chart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved graphs to: {fig_dir}")
    
    def create_summary_report(self, results: Dict, df: pd.DataFrame):
        """Create comprehensive summary report"""
        report_path = self.results_dir / f'benchmark_report_{self.timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write("# Benchmark Results Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            if results.get('average'):
                avg = results['average']
                f.write(f"- **Average MOTA:** {avg.get('mota', 0):.3f}\n")
                f.write(f"- **Average MOTP:** {avg.get('motp', 0):.3f}\n")
                f.write(f"- **Average IDF1:** {avg.get('idf1', 0):.3f}\n")
                f.write(f"- **Average Precision:** {avg.get('precision', 0):.3f}\n")
                f.write(f"- **Average Recall:** {avg.get('recall', 0):.3f}\n")
                f.write(f"- **Total ID Switches:** {avg.get('id_switches', 0)}\n\n")
            
            f.write("## Per-Sequence Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Performance Analysis\n\n")
            f.write("### Key Achievements\n\n")
            if results.get('average'):
                avg = results['average']
                mota = avg.get('mota', 0)
                if mota > 0.7:
                    f.write(f"- ✅ **Excellent MOTA ({mota:.1%})** - Top-tier performance\n")
                elif mota > 0.6:
                    f.write(f"- ✅ **Good MOTA ({mota:.1%})** - Competitive performance\n")
                else:
                    f.write(f"- ⚠️ **MOTA ({mota:.1%})** - Room for improvement\n")
            
            f.write("\n### Comparison with Baselines\n\n")
            f.write("| Method | MOTA | Notes |\n")
            f.write("|--------|------|-------|\n")
            f.write("| Simple Tracker | ~0.40-0.50 | Basic centroid tracking |\n")
            f.write("| DeepSORT | ~0.60-0.70 | Standard baseline |\n")
            f.write("| Our System | - | See results above |\n")
            f.write("| SOTA | ~0.75-0.85 | State-of-the-art methods |\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- Comparison table (CSV, Markdown, LaTeX)\n")
            f.write("- Performance graphs (PNG)\n")
            f.write("- This report\n\n")
        
        print(f"Saved report: {report_path}")


def main():
    """Main function to collect and organize results"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect and visualize benchmark results')
    parser.add_argument('--summary', type=str, 
                       default='outputs/mot_results/summary.json',
                       help='Path to benchmark summary JSON')
    parser.add_argument('--output-dir', type=str,
                       default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    collector = ResultsCollector(results_dir=args.output_dir)
    
    # Load results
    if Path(args.summary).exists():
        print(f"Loading results from: {args.summary}")
        results = collector.load_benchmark_results(args.summary)
        
        # Create comparison table
        print("Creating comparison table...")
        df = collector.create_comparison_table(results)
        collector.save_comparison_table(df, format='both')
        
        # Create graphs
        print("Creating performance graphs...")
        collector.create_performance_graphs(results)
        
        # Create report
        print("Creating summary report...")
        collector.create_summary_report(results, df)
        
        print(f"\n✅ All results collected in: {collector.results_dir}")
    else:
        print(f"⚠️  Results file not found: {args.summary}")
        print("Run benchmark first: python scripts/evaluate_benchmark.py")


if __name__ == '__main__':
    main()

