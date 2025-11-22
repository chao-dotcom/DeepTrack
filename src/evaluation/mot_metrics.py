"""MOT evaluation metrics implementation"""
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False
    print("Warning: motmetrics not available. Install with: pip install motmetrics")


class MOTEvaluator:
    """
    Comprehensive MOT evaluation using standard metrics
    Calculates MOTA, MOTP, IDF1, etc.
    """
    def __init__(self):
        if MOTMETRICS_AVAILABLE:
            self.acc = mm.MOTAccumulator(auto_id=True)
        else:
            self.acc = None
            print("Warning: Using simplified evaluation (install motmetrics for full metrics)")
    
    def update(self, gt_tracks: List, pred_tracks: List, frame_id: int):
        """
        Update accumulator with frame results
        
        Args:
            gt_tracks: List of ground truth [x1, y1, x2, y2, track_id]
            pred_tracks: List of predictions [x1, y1, x2, y2, track_id]
            frame_id: Current frame number
        """
        if not MOTMETRICS_AVAILABLE:
            return
        
        if len(gt_tracks) == 0 and len(pred_tracks) == 0:
            return
        
        # Extract IDs and boxes
        gt_ids = [int(t[4]) for t in gt_tracks] if len(gt_tracks) > 0 else []
        pred_ids = [int(t[4]) for t in pred_tracks] if len(pred_tracks) > 0 else []
        
        gt_boxes = np.array([t[:4] for t in gt_tracks]) if len(gt_tracks) > 0 else np.empty((0, 4))
        pred_boxes = np.array([t[:4] for t in pred_tracks]) if len(pred_tracks) > 0 else np.empty((0, 4))
        
        # Compute distance matrix (IoU-based)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)))
        
        # Update accumulator
        self.acc.update(gt_ids, pred_ids, distances, frameid=frame_id)
    
    def compute_metrics(self) -> Dict:
        """Compute final MOT metrics"""
        if not MOTMETRICS_AVAILABLE or self.acc is None:
            return self._compute_simplified_metrics()
        
        mh = mm.metrics.create()
        
        summary = mh.compute(
            self.acc,
            metrics=[
                'num_frames',
                'num_matches',
                'num_switches',
                'num_false_positives',
                'num_misses',
                'mota',
                'motp',
                'precision',
                'recall'
            ],
            name='Overall'
        )
        
        return summary
    
    def _compute_simplified_metrics(self) -> Dict:
        """Simplified metrics without motmetrics"""
        return {
            'mota': {'Overall': 0.0},
            'motp': {'Overall': 0.0},
            'precision': {'Overall': 0.0},
            'recall': {'Overall': 0.0},
            'num_switches': {'Overall': 0},
            'num_false_positives': {'Overall': 0},
            'num_misses': {'Overall': 0}
        }
    
    def compute_idf1(self, gt_file: str, pred_file: str) -> float:
        """
        Compute IDF1 score (ID F1 score)
        Measures identity preservation quality
        """
        # Load ground truth
        gt_data = self._load_mot_format(gt_file)
        
        # Load predictions
        pred_data = self._load_mot_format(pred_file)
        
        # Compute IDF1
        idf1_score = self._calculate_idf1(gt_data, pred_data)
        
        return idf1_score
    
    def _load_mot_format(self, filepath: str) -> Dict[int, List[Dict]]:
        """Load MOT format tracking file"""
        data = defaultdict(list)
        
        if not Path(filepath).exists():
            return data
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                try:
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])
                    
                    data[frame].append({
                        'id': track_id,
                        'bbox': [x, y, x+w, y+h]
                    })
                except (ValueError, IndexError):
                    continue
        
        return data
    
    def _calculate_idf1(self, gt_data: Dict, pred_data: Dict) -> float:
        """Calculate IDF1 metric"""
        idtp = 0  # ID true positives
        idfp = 0  # ID false positives
        idfn = 0  # ID false negatives
        
        all_frames = set(gt_data.keys()) | set(pred_data.keys())
        
        for frame in all_frames:
            if frame not in pred_data:
                if frame in gt_data:
                    idfn += len(gt_data[frame])
                continue
            
            if frame not in gt_data:
                idfp += len(pred_data[frame])
                continue
            
            gt_tracks = gt_data[frame]
            pred_tracks = pred_data[frame]
            
            # Match tracks based on IoU
            matched_gt = set()
            matched_pred = set()
            
            for i, gt in enumerate(gt_tracks):
                best_iou = 0
                best_j = -1
                
                for j, pred in enumerate(pred_tracks):
                    if j in matched_pred:
                        continue
                    
                    iou = self._compute_iou(gt['bbox'], pred['bbox'])
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_j = j
                
                if best_j != -1:
                    # Check if IDs match
                    if gt['id'] == pred_tracks[best_j]['id']:
                        idtp += 1
                    else:
                        idfp += 1
                    
                    matched_gt.add(i)
                    matched_pred.add(best_j)
                else:
                    idfn += 1
            
            # Unmatched predictions are false positives
            idfp += len(pred_tracks) - len(matched_pred)
        
        # Calculate IDF1
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (idtp + idfp + idfn) > 0 else 0.0
        
        return idf1
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class MOTBenchmark:
    """
    Run full MOT benchmark evaluation
    """
    def __init__(self, tracker, gt_path: str, output_path: str):
        self.tracker = tracker
        self.gt_path = Path(gt_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def evaluate_sequence(self, sequence_name: str) -> Dict:
        """Evaluate single MOT sequence"""
        print(f"Evaluating {sequence_name}...")
        
        seq_path = self.gt_path / sequence_name
        img_dir = seq_path / 'img1'
        gt_file = seq_path / 'gt' / 'gt.txt'
        
        if not img_dir.exists() or not gt_file.exists():
            print(f"Warning: {sequence_name} missing required files")
            return {}
        
        # Run tracker
        try:
            results = self.tracker.process_video(
                video_path=str(img_dir),
                output_path=None,
                visualize=False
            )
        except Exception as e:
            print(f"Error processing {sequence_name}: {e}")
            return {}
        
        # Convert results to frame-based format
        frame_results = {}
        for frame_data in results.get('tracks', []):
            frame_id = frame_data['frame']
            tracks = []
            for det in frame_data['detections']:
                bbox = det['bbox']
                track_id = det['track_id']
                tracks.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
            frame_results[frame_id] = tracks
        
        # Save results in MOT format
        result_file = self.output_path / f'{sequence_name}.txt'
        self._save_mot_format(frame_results, result_file)
        
        # Evaluate
        evaluator = MOTEvaluator()
        
        # Load ground truth
        gt_data = self._load_gt(gt_file)
        
        # Compare frame by frame
        all_frames = set(gt_data.keys()) | set(frame_results.keys())
        for frame_id in sorted(all_frames):
            gt_tracks = gt_data.get(frame_id, [])
            pred_tracks = frame_results.get(frame_id, [])
            evaluator.update(gt_tracks, pred_tracks, frame_id)
        
        # Compute metrics
        metrics = evaluator.compute_metrics()
        idf1 = evaluator.compute_idf1(str(gt_file), str(result_file))
        
        # Extract values
        if MOTMETRICS_AVAILABLE and hasattr(metrics, 'values'):
            mota = metrics['mota'].values[0] if 'mota' in metrics else 0.0
            motp = metrics['motp'].values[0] if 'motp' in metrics else 0.0
            precision = metrics['precision'].values[0] if 'precision' in metrics else 0.0
            recall = metrics['recall'].values[0] if 'recall' in metrics else 0.0
            id_switches = metrics['num_switches'].values[0] if 'num_switches' in metrics else 0
        else:
            mota = metrics.get('mota', {}).get('Overall', 0.0)
            motp = metrics.get('motp', {}).get('Overall', 0.0)
            precision = metrics.get('precision', {}).get('Overall', 0.0)
            recall = metrics.get('recall', {}).get('Overall', 0.0)
            id_switches = metrics.get('num_switches', {}).get('Overall', 0)
        
        print(f"\nResults for {sequence_name}:")
        print(f"MOTA: {mota:.3f}")
        print(f"MOTP: {motp:.3f}")
        print(f"IDF1: {idf1:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"ID Switches: {id_switches}")
        
        return {
            'mota': float(mota),
            'motp': float(motp),
            'idf1': float(idf1),
            'precision': float(precision),
            'recall': float(recall),
            'id_switches': int(id_switches)
        }
    
    def evaluate_all(self) -> Tuple[Dict, Dict]:
        """Evaluate all sequences"""
        if not self.gt_path.exists():
            print(f"Error: Ground truth path {self.gt_path} does not exist")
            return {}, {}
        
        sequences = [d.name for d in self.gt_path.iterdir() if d.is_dir()]
        
        if not sequences:
            print(f"No sequences found in {self.gt_path}")
            return {}, {}
        
        all_results = {}
        for seq in sequences:
            results = self.evaluate_sequence(seq)
            if results:
                all_results[seq] = results
        
        if not all_results:
            return {}, {}
        
        # Compute average
        avg_results = {
            'mota': np.mean([r['mota'] for r in all_results.values()]),
            'motp': np.mean([r['motp'] for r in all_results.values()]),
            'idf1': np.mean([r['idf1'] for r in all_results.values()]),
            'precision': np.mean([r['precision'] for r in all_results.values()]),
            'recall': np.mean([r['recall'] for r in all_results.values()])
        }
        
        print("\n" + "="*50)
        print("OVERALL RESULTS")
        print("="*50)
        print(f"Average MOTA: {avg_results['mota']:.3f}")
        print(f"Average MOTP: {avg_results['motp']:.3f}")
        print(f"Average IDF1: {avg_results['idf1']:.3f}")
        print(f"Average Precision: {avg_results['precision']:.3f}")
        print(f"Average Recall: {avg_results['recall']:.3f}")
        
        return all_results, avg_results
    
    def _load_gt(self, gt_file: Path) -> Dict[int, List]:
        """Load ground truth annotations"""
        gt_data = defaultdict(list)
        
        if not gt_file.exists():
            return gt_data
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                try:
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    
                    if conf > 0:  # Only consider valid annotations
                        gt_data[frame].append([x, y, x+w, y+h, track_id])
                except (ValueError, IndexError):
                    continue
        
        return gt_data
    
    def _save_mot_format(self, results: Dict[int, List], output_file: Path):
        """Save tracking results in MOT format"""
        with open(output_file, 'w') as f:
            for frame_id in sorted(results.keys()):
                tracks = results[frame_id]
                for track in tracks:
                    x1, y1, x2, y2, track_id = track
                    w = x2 - x1
                    h = y2 - y1
                    # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
                    f.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")


# Usage example
if __name__ == '__main__':
    from src.inference.deepsort_tracker import DeepSORTVideoTracker
    
    # Initialize tracker
    config = {
        'detection': {'conf_threshold': 0.25},
        'tracking': {
            'max_dist': 0.2,
            'max_iou_distance': 0.7,
            'max_age': 30,
            'n_init': 3
        }
    }
    
    tracker = DeepSORTVideoTracker(
        detection_model_path='models/checkpoints/yolov8n.pt',
        reid_model_path=None,  # Optional
        config=config
    )
    
    # Run benchmark
    benchmark = MOTBenchmark(
        tracker=tracker,
        gt_path='data/raw/MOT20/MOT20/train',
        output_path='outputs/mot_results'
    )
    
    all_results, avg_results = benchmark.evaluate_all()


