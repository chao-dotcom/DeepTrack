"""Calculate real MOT metrics from tracking results and ground truth"""
import sys
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

# Ensure UTF-8 output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_ground_truth(gt_path):
    """Load MOT format ground truth"""
    gt_file = Path(gt_path) / "gt" / "gt.txt"
    if not gt_file.exists():
        return None
    
    gt_data = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                visibility = float(parts[8])
                
                # Only include visible objects
                if visibility > 0.5:
                    gt_data.append({
                        'frame': frame_id,
                        'id': track_id,
                        'bbox': [x, y, w, h],
                        'conf': conf
                    })
    
    return gt_data

def load_tracking_results(json_path):
    """Load tracking results from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tracks = []
    for frame_data in data.get('tracks', []):
        frame_id = frame_data.get('frame', 0) + 1  # MOT format is 1-indexed
        for det in frame_data.get('detections', []):
            track_id = det.get('track_id', -1)
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                tracks.append({
                    'frame': frame_id,
                    'id': track_id,
                    'bbox': [x1, y1, w, h]
                })
    
    return tracks

def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to [x1, y1, x2, y2]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    # Calculate intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def calculate_mot_metrics(gt_data, track_data):
    """Calculate MOT metrics"""
    # Group by frame
    gt_by_frame = defaultdict(list)
    track_by_frame = defaultdict(list)
    
    for item in gt_data:
        gt_by_frame[item['frame']].append(item)
    
    for item in track_data:
        track_by_frame[item['frame']].append(item)
    
    # Calculate metrics
    total_frames = max(len(gt_by_frame), len(track_by_frame))
    total_gt = len(gt_data)
    total_track = len(track_data)
    
    # Match detections
    matches = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    total_iou = 0
    
    # Track ID mapping
    id_mapping = {}  # gt_id -> track_id
    prev_mapping = {}
    
    for frame_id in sorted(set(list(gt_by_frame.keys()) + list(track_by_frame.keys()))):
        gt_objects = gt_by_frame.get(frame_id, [])
        track_objects = track_by_frame.get(frame_id, [])
        
        # Build cost matrix
        cost_matrix = []
        gt_indices = []
        track_indices = []
        
        for i, gt_obj in enumerate(gt_objects):
            for j, track_obj in enumerate(track_objects):
                iou = calculate_iou(gt_obj['bbox'], track_obj['bbox'])
                if iou > 0.5:  # Threshold
                    cost = 1 - iou
                    cost_matrix.append((i, j, cost, iou))
                    if i not in gt_indices:
                        gt_indices.append(i)
                    if j not in track_indices:
                        track_indices.append(j)
        
        # Simple greedy matching
        matched_gt = set()
        matched_track = set()
        frame_matches = []
        
        # Sort by IoU (descending)
        cost_matrix.sort(key=lambda x: x[3], reverse=True)
        
        for i, j, cost, iou in cost_matrix:
            if i not in matched_gt and j not in matched_track:
                matched_gt.add(i)
                matched_track.add(j)
                frame_matches.append((i, j, iou))
                
                gt_id = gt_objects[i]['id']
                track_id = track_objects[j]['id']
                
                # Check for ID switch
                if gt_id in prev_mapping and prev_mapping[gt_id] != track_id:
                    id_switches += 1
                
                id_mapping[gt_id] = track_id
                matches += 1
                total_iou += iou
        
        # False positives (tracked but not in GT)
        false_positives += len(track_objects) - len(matched_track)
        
        # False negatives (in GT but not tracked)
        false_negatives += len(gt_objects) - len(matched_gt)
        
        prev_mapping = id_mapping.copy()
    
    # Calculate metrics
    if total_gt == 0:
        return None
    
    mota = 1 - (false_positives + false_negatives + id_switches) / total_gt
    motp = total_iou / matches if matches > 0 else 0
    precision = matches / (matches + false_positives) if (matches + false_positives) > 0 else 0
    recall = matches / (matches + false_negatives) if (matches + false_negatives) > 0 else 0
    
    # IDF1 approximation (simplified)
    idf1 = matches / (matches + 0.5 * (false_positives + false_negatives)) if matches > 0 else 0
    
    return {
        'mota': max(0, mota),
        'motp': motp,
        'precision': precision,
        'recall': recall,
        'idf1': idf1,
        'id_switches': id_switches,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'matches': matches,
        'total_gt': total_gt,
        'total_track': total_track,
        'total_frames': total_frames
    }

def main():
    """Calculate metrics for a sequence"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate real MOT metrics')
    parser.add_argument('--gt-path', type=str, required=True,
                       help='Path to ground truth directory (e.g., data/raw/MOT20/MOT20/train/MOT20-01)')
    parser.add_argument('--results-json', type=str, required=True,
                       help='Path to tracking results JSON')
    parser.add_argument('--output', type=str,
                       help='Path to save metrics JSON')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Calculating Real MOT Metrics")
    print("="*70)
    
    # Load data
    print(f"\n▶ Loading ground truth: {args.gt_path}")
    gt_data = load_ground_truth(args.gt_path)
    if not gt_data:
        print("✗ Ground truth not found!")
        return
    
    print(f"  ✓ Loaded {len(gt_data)} ground truth objects")
    
    print(f"\n▶ Loading tracking results: {args.results_json}")
    track_data = load_tracking_results(args.results_json)
    print(f"  ✓ Loaded {len(track_data)} tracked objects")
    
    # Calculate metrics
    print(f"\n▶ Calculating metrics...")
    metrics = calculate_mot_metrics(gt_data, track_data)
    
    if metrics:
        print(f"\n{'='*70}")
        print("Results:")
        print(f"{'='*70}")
        print(f"MOTA:  {metrics['mota']:.4f} ({metrics['mota']*100:.2f}%)")
        print(f"MOTP:  {metrics['motp']:.4f} ({metrics['motp']*100:.2f}%)")
        print(f"IDF1:  {metrics['idf1']:.4f} ({metrics['idf1']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"ID Switches: {metrics['id_switches']}")
        print(f"Matches: {metrics['matches']} / {metrics['total_gt']} GT objects")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        
        # Save if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✓ Metrics saved to: {args.output}")
    else:
        print("✗ Failed to calculate metrics")

if __name__ == "__main__":
    main()

