"""Test ID consistency improvements"""
import sys
import json
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def analyze_id_consistency(json_path):
    """Analyze ID consistency from tracking results"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track ID changes per object (simplified analysis)
    # Count how many times each bbox position gets different IDs
    frame_tracks = data.get('tracks', [])
    
    # Build trajectory for each ID
    id_trajectories = {}
    for frame_data in frame_tracks:
        frame_id = frame_data.get('frame', 0)
        for det in frame_data.get('detections', []):
            track_id = det.get('track_id', -1)
            bbox = det.get('bbox', [])
            
            if track_id not in id_trajectories:
                id_trajectories[track_id] = []
            
            if len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                id_trajectories[track_id].append({
                    'frame': frame_id,
                    'center': (cx, cy)
                })
    
    # Calculate statistics
    track_lengths = [len(traj) for traj in id_trajectories.values()]
    
    if not track_lengths:
        return None
    
    stats = {
        'total_tracks': len(id_trajectories),
        'avg_length': sum(track_lengths) / len(track_lengths),
        'max_length': max(track_lengths),
        'min_length': min(track_lengths),
        'long_tracks': len([l for l in track_lengths if l >= 20]),  # Tracks >= 20 frames
        'very_long_tracks': len([l for l in track_lengths if l >= 50]),  # Tracks >= 50 frames
    }
    
    return stats

def main():
    """Test and compare ID consistency"""
    print("="*70)
    print("ID Consistency Test - Enhanced Settings")
    print("="*70)
    
    # Run tracking with enhanced settings
    print("\n▶ Running tracking with enhanced ID consistency settings...")
    print("   - n_init: 7 (more frames before confirmation)")
    print("   - max_dist: 0.25 (tighter matching)")
    print("   - Appearance weight: 80% (stronger appearance matching)")
    print("   - Confirmed track threshold: 0.5x (very strict)")
    print("   - Feature gallery: 200 features (more history)")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = f"data/processed/real_benchmark_results/MOT20-01_enhanced_{timestamp}.json"
    output_video = f"data/processed/real_benchmark_results/MOT20-01_enhanced_{timestamp}.mp4"
    
    import subprocess
    cmd = [
        sys.executable, "-m", "src.inference.deepsort_tracker",
        "--input", "data/raw/MOT20/MOT20/train/MOT20-01/img1",
        "--output", output_video,
        "--detection-model", "models/checkpoints/yolov8n.pt"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    if result.returncode == 0:
        print("\n✅ Tracking completed!")
        
        # Find the actual JSON file (it might have a different timestamp)
        json_files = list(Path("data/processed/real_benchmark_results").glob("MOT20-01_enhanced_*.json"))
        if json_files:
            json_path = json_files[0]
            print(f"\n📊 Analyzing ID consistency: {json_path.name}")
            
            stats = analyze_id_consistency(json_path)
            if stats:
                print(f"\n{'Metric':<30} {'Value':<20}")
                print("-"*50)
                print(f"{'Total Tracks':<30} {stats['total_tracks']:<20}")
                print(f"{'Avg Track Length':<30} {stats['avg_length']:.2f} frames")
                print(f"{'Max Track Length':<30} {stats['max_length']:<20} frames")
                print(f"{'Long Tracks (>=20 frames)':<30} {stats['long_tracks']:<20}")
                print(f"{'Very Long Tracks (>=50 frames)':<30} {stats['very_long_tracks']:<20}")
                
                # Compare with previous
                prev_json = Path("data/processed/real_benchmark_results/MOT20-01_fixed_20251122_102837.json")
                if prev_json.exists():
                    prev_stats = analyze_id_consistency(prev_json)
                    if prev_stats:
                        print("\n" + "="*70)
                        print("Comparison with Previous Version:")
                        print("="*70)
                        print(f"\n{'Metric':<30} {'Previous':<15} {'Enhanced':<15} {'Change':<15}")
                        print("-"*70)
                        
                        metrics = [
                            ('Avg Track Length', 'avg_length', 'frames'),
                            ('Max Track Length', 'max_length', 'frames'),
                            ('Long Tracks (>=20)', 'long_tracks', ''),
                            ('Very Long Tracks (>=50)', 'very_long_tracks', ''),
                        ]
                        
                        for name, key, unit in metrics:
                            prev_val = prev_stats[key]
                            new_val = stats[key]
                            if unit == 'frames':
                                change = new_val - prev_val
                                change_str = f"{change:+.2f} {unit}"
                            else:
                                change = new_val - prev_val
                                change_str = f"{change:+.0f}"
                            
                            prev_str = f"{prev_val:.2f} {unit}" if unit else f"{prev_val}"
                            new_str = f"{new_val:.2f} {unit}" if unit else f"{new_val}"
                            
                            print(f"{name:<30} {prev_str:<15} {new_str:<15} {change_str:<15}")
        else:
            print("⚠ JSON file not found")
    else:
        print(f"\n✗ Tracking failed: {result.stderr[:500]}")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)

if __name__ == "__main__":
    main()

