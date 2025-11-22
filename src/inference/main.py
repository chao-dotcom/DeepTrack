"""Main inference entry point"""
import argparse
import cv2
import sys
import os
import json
from pathlib import Path
from typing import Union, Tuple, Optional
from datetime import datetime

# Import detection and tracking modules
try:
    from src.models.detection.yolo_detector import YOLODetector
    from src.models.tracking.simple_tracker import SimpleTracker
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    print("Warning: Detection modules not available. Running in pass-through mode.")


def is_image_sequence(path: str) -> bool:
    """Check if path is a directory with image files"""
    path_obj = Path(path)
    if not path_obj.is_dir():
        return False
    
    # Check if directory contains image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = list(path_obj.glob('*'))
    image_files = [f for f in files if f.suffix.lower() in image_extensions]
    return len(image_files) > 0


def get_image_sequence_fps(seq_path: str) -> float:
    """Try to get FPS from seqinfo.ini (MOT format) or use default"""
    seq_path_obj = Path(seq_path)
    seqinfo_file = seq_path_obj.parent / 'seqinfo.ini'
    
    if seqinfo_file.exists():
        with open(seqinfo_file, 'r') as f:
            for line in f:
                if line.startswith('frameRate='):
                    return float(line.split('=')[1].strip())
    
    return 30.0  # Default FPS


def create_image_sequence_capture(img_dir: str) -> Tuple[list, float, Tuple[int, int]]:
    """Create a frame generator from image sequence"""
    img_path = Path(img_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Get all image files sorted
    image_files = sorted([f for f in img_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions])
    
    if not image_files:
        raise ValueError(f"No image files found in {img_dir}")
    
    # Get FPS
    fps = get_image_sequence_fps(img_dir)
    
    # Get dimensions from first image
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width = first_img.shape[:2]
    
    return image_files, fps, (width, height)


def main():
    """Main entry point for inference"""
    parser = argparse.ArgumentParser(description="People Tracking Inference")
    parser.add_argument("--input", required=True, 
                       help="Input video file, camera index (0 for webcam), or image sequence directory")
    parser.add_argument("--output", help="Output video file path (optional)")
    parser.add_argument("--model", help="Path to model checkpoint")
    parser.add_argument("--display", action="store_true", help="Display video output")
    
    args = parser.parse_args()
    
    # Determine input source
    is_camera = False
    is_image_seq = False
    image_files = None
    fps = 30.0
    width, height = 1920, 1080
    
    if args.input.isdigit():
        input_source = int(args.input)
        is_camera = True
        print(f"Using camera {input_source}")
    elif is_image_sequence(args.input):
        is_image_seq = True
        print(f"Processing image sequence: {args.input}")
        try:
            image_files, fps, (width, height) = create_image_sequence_capture(args.input)
            print(f"Found {len(image_files)} images, {width}x{height} @ {fps} FPS")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            sys.exit(1)
        input_source = args.input
        print(f"Processing video: {args.input}")
    
    # Open video source (or prepare image sequence)
    if is_image_seq:
        cap = None  # We'll handle frames manually
        frame_index = 0
    else:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{args.input}'")
            sys.exit(1)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Initialize detection and tracking
    detector = None
    tracker = None
    use_detection = False
    
    if DETECTION_AVAILABLE:
        model_path = args.model or 'yolov8n.pt'
        # Check if model exists locally
        if not os.path.exists(model_path):
            # Try in checkpoints directory
            checkpoint_path = f"models/checkpoints/{os.path.basename(model_path)}"
            if os.path.exists(checkpoint_path):
                model_path = checkpoint_path
            else:
                print(f"Model not found at {model_path}, using default (will auto-download)")
                model_path = 'yolov8n.pt'
        
        try:
            detector = YOLODetector(model_path=model_path, confidence_threshold=0.5)
            tracker = SimpleTracker(max_disappeared=30, max_distance=100.0)
            use_detection = True
            print("✓ Detection and tracking enabled")
        except Exception as e:
            print(f"Warning: Could not initialize detector: {e}")
            print("Running in pass-through mode (no detection)")
    
    # Setup output video writer if output path is provided
    out = None
    if args.output:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output will be saved to: {args.output}")
    
    # Statistics tracking
    frame_count = 0
    total_detections = 0
    max_tracks = 0
    detection_history = []
    
    try:
        while True:
            # Read frame
            if is_image_seq:
                if frame_index >= len(image_files):
                    break
                frame = cv2.imread(str(image_files[frame_index]))
                if frame is None:
                    print(f"Warning: Could not read {image_files[frame_index]}")
                    frame_index += 1
                    continue
                ret = True
                frame_index += 1
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_count += 1
            
            # Process frame with detection and tracking
            if use_detection and detector and tracker:
                # Detect people
                detections = detector.detect(frame)
                total_detections += len(detections)
                
                # Update tracker
                tracked_objects = tracker.update(detections)
                current_tracks = tracker.get_track_count()
                max_tracks = max(max_tracks, current_tracks)
                
                # Draw detections and tracks
                processed_frame = detector.draw_detections(frame, tracked_objects)
                
                # Draw track IDs and trajectories
                for obj in tracked_objects:
                    if 'id' in obj:
                        x1, y1, x2, y2 = [int(coord) for coord in obj['bbox']]
                        track_id = obj['id']
                        
                        # Draw track ID
                        cv2.putText(processed_frame, f"ID: {track_id}", 
                                   (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (255, 0, 255), 2)
                
                # Draw statistics on frame
                stats_text = [
                    f"Frame: {frame_count}",
                    f"Detections: {len(detections)}",
                    f"Active Tracks: {current_tracks}",
                    f"Total Detections: {total_detections}"
                ]
                y_offset = 30
                for text in stats_text:
                    cv2.putText(processed_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 25
                
                # Store detection info
                detection_history.append({
                    'frame': frame_count,
                    'detections': len(detections),
                    'tracks': current_tracks
                })
            else:
                # Pass-through mode
                processed_frame = frame
            
            # Write frame if output is specified
            if out:
                out.write(processed_frame)
            
            # Display frame if requested
            if args.display:
                cv2.imshow('People Tracking', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 30 == 0:
                if use_detection:
                    print(f"Processed {frame_count} frames... Detections: {total_detections}, Max Tracks: {max_tracks}")
                else:
                    print(f"Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        if cap is not None:
            cap.release()
        if out:
            out.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        if use_detection:
            print(f"Total detections: {total_detections}")
            print(f"Average detections per frame: {total_detections/frame_count:.2f}" if frame_count > 0 else "N/A")
            print(f"Maximum concurrent tracks: {max_tracks}")
        print("="*60)
        
        # Save results JSON if output video was specified
        if args.output and use_detection:
            results_path = Path(args.output).with_suffix('.json')
            results_data = {
                'input': args.input,
                'output': args.output,
                'timestamp': datetime.now().isoformat(),
                'total_frames': frame_count,
                'total_detections': total_detections,
                'max_tracks': max_tracks,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'detection_history': detection_history
            }
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

