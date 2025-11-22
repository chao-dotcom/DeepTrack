"""Script to run MOT benchmark evaluation"""
import argparse
from pathlib import Path
import yaml

from src.evaluation.mot_metrics import MOTBenchmark
from src.inference.deepsort_tracker import DeepSORTVideoTracker


def main():
    parser = argparse.ArgumentParser(description='Run MOT benchmark evaluation')
    parser.add_argument('--gt-path', type=str, 
                       default='data/raw/MOT20/MOT20/train',
                       help='Path to ground truth sequences')
    parser.add_argument('--output-path', type=str,
                       default='outputs/mot_results',
                       help='Path to save evaluation results')
    parser.add_argument('--detection-model', type=str,
                       default='models/checkpoints/yolov8n.pt',
                       help='Path to detection model')
    parser.add_argument('--reid-model', type=str,
                       help='Path to ReID model (optional)')
    parser.add_argument('--config', type=str,
                       help='Path to tracking config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'detection': {'conf_threshold': 0.25},
            'tracking': {
                'max_dist': 0.2,
                'max_iou_distance': 0.7,
                'max_age': 30,
                'n_init': 3
            }
        }
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = DeepSORTVideoTracker(
        detection_model_path=args.detection_model,
        reid_model_path=args.reid_model,
        config=config
    )
    
    # Run benchmark
    print(f"\nRunning benchmark on: {args.gt_path}")
    benchmark = MOTBenchmark(
        tracker=tracker,
        gt_path=args.gt_path,
        output_path=args.output_path
    )
    
    all_results, avg_results = benchmark.evaluate_all()
    
    # Save summary
    if avg_results:
        summary_path = Path(args.output_path) / 'summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump({
                'per_sequence': all_results,
                'average': avg_results
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()


