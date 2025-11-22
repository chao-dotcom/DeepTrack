"""DeepSORT video tracker integration"""
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict, List

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Detection will not work.")

from src.models.tracking.deepsort import DeepSORT
from src.models.reid.reid_model import ReIDModel


class DeepSORTVideoTracker:
    """
    Video tracking using trained detection and ReID models
    """
    def __init__(self, detection_model_path: str, reid_model_path: Optional[str] = None, 
                 config: Optional[Dict] = None):
        self.config = config or {
            'detection': {
                'conf_threshold': 0.15,  # Lower threshold
                'iou_threshold': 0.45   # Add NMS
            },
            'tracking': {
                'max_dist': 0.32,  # Increase for more flexibility
                'max_iou_distance': 0.7,
                'max_age': 50,  # Increased for better persistence
                'n_init': 5  # Reduce from 7 to 5 for faster confirmation
            }
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load detection model
        if YOLO_AVAILABLE:
            try:
                self.detector = YOLO(detection_model_path)
                print(f"Loaded detection model: {detection_model_path}")
            except Exception as e:
                print(f"Error loading detection model: {e}")
                self.detector = None
        else:
            self.detector = None
            print("Warning: YOLO not available. Detection disabled.")
        
        # Load ReID model if provided
        self.reid_model = None
        if reid_model_path and Path(reid_model_path).exists():
            try:
                checkpoint = torch.load(reid_model_path, map_location=self.device)
                num_classes = checkpoint.get('num_classes', 751)
                
                self.reid_model = ReIDModel(
                    num_classes=num_classes,
                    feature_dim=2048
                ).to(self.device)
                
                self.reid_model.load_state_dict(checkpoint['model_state_dict'])
                self.reid_model.eval()
                print(f"Loaded ReID model: {reid_model_path}")
            except Exception as e:
                print(f"Error loading ReID model: {e}")
                print("Continuing without ReID features...")
        
        # Initialize DeepSORT
        self.tracker = DeepSORT(
            reid_model=self.reid_model,
            max_dist=self.config['tracking']['max_dist'],
            max_iou_distance=self.config['tracking']['max_iou_distance'],
            max_age=self.config['tracking']['max_age'],
            n_init=self.config['tracking']['n_init']
        )
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     visualize: bool = True) -> Dict:
        """
        Process video and track people
        
        Args:
            video_path: Path to input video, image sequence directory, or webcam index (0, 1, etc.)
            output_path: Path to save output video (optional)
            visualize: Whether to draw bounding boxes
        
        Returns:
            tracking_results: Dictionary with frame-by-frame tracking info
        """
        # Check if it's a webcam index (numeric string)
        is_webcam = False
        webcam_index = None
        try:
            webcam_index = int(video_path)
            is_webcam = True
        except ValueError:
            pass
        
        # Check if it's an image sequence
        if not is_webcam:
            video_path_obj = Path(video_path)
            if video_path_obj.is_dir():
                return self._process_image_sequence(video_path, output_path, visualize)
        
        # Process video file or webcam
        if is_webcam:
            cap = cv2.VideoCapture(webcam_index)
            print(f"Opening webcam {webcam_index}...")
        else:
            cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video/webcam: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # For webcam, total_frames might be 0
        if is_webcam:
            total_frames = 0  # Infinite stream
            print("Press 'q' to quit")
        
        # Video writer
        writer = None
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Tracking results
        tracking_results = {
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames if total_frames > 0 else 'streaming'
            },
            'tracks': []
        }
        
        frame_idx = 0
        if is_webcam:
            pbar = tqdm(desc='Processing webcam stream', unit='frames')
        else:
            pbar = tqdm(total=total_frames, desc='Processing video')
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect people
                detections = self._detect_people(frame)
                
                # Update tracker
                tracks = self.tracker.update(detections, frame)
                
                # Store results
                frame_tracks = []
                for track in tracks:
                    x1, y1, x2, y2, track_id = track
                    frame_tracks.append({
                        'track_id': int(track_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
                
                tracking_results['tracks'].append({
                    'frame': frame_idx,
                    'detections': frame_tracks
                })
                
                # Visualize
                if visualize:
                    vis_frame = self._visualize_tracks(frame.copy(), tracks)
                    
                    if writer:
                        writer.write(vis_frame)
                    
                    # Show window for webcam or if no output file specified
                    if is_webcam or not output_path:
                        cv2.imshow('People Tracking', vis_frame)
                        # Check for 'q' key to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nStopped by user (pressed 'q')")
                            break
                
                frame_idx += 1
                pbar.update(1)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\n\nError during processing: {e}")
            raise
        finally:
            # Always cleanup resources, even if an error occurred
            pbar.close()
            if cap.isOpened():
                cap.release()
                print("Camera released")
            if writer:
                writer.release()
                print("Video writer released")
            if visualize and (is_webcam or not output_path):
                cv2.destroyAllWindows()
                print("Windows closed")
        
        # Calculate statistics
        tracking_results['statistics'] = self._calculate_statistics(tracking_results)
        
        return tracking_results
    
    def _process_image_sequence(self, img_dir: str, output_path: Optional[str] = None,
                                visualize: bool = True) -> Dict:
        """Process image sequence (MOT format)"""
        img_dir_path = Path(img_dir)
        image_files = sorted(img_dir_path.glob('*.jpg'))
        
        if not image_files:
            raise ValueError(f"No images found in {img_dir}")
        
        # Get image dimensions from first frame
        first_img = cv2.imread(str(image_files[0]))
        height, width = first_img.shape[:2]
        fps = 25  # Default FPS
        
        # Video writer
        writer = None
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        tracking_results = {
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': len(image_files)
            },
            'tracks': []
        }
        
        pbar = tqdm(image_files, desc='Processing images')
        
        for frame_idx, img_path in enumerate(pbar):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Detect people
            detections = self._detect_people(frame)
            
            # Update tracker
            tracks = self.tracker.update(detections, frame)
            
            # Store results
            frame_tracks = []
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                frame_tracks.append({
                    'track_id': int(track_id),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
            
            tracking_results['tracks'].append({
                'frame': frame_idx,
                'detections': frame_tracks
            })
            
            # Visualize
            if visualize:
                vis_frame = self._visualize_tracks(frame.copy(), tracks)
                
                if writer:
                    writer.write(vis_frame)
        
        pbar.close()
        if writer:
            writer.release()
        
        tracking_results['statistics'] = self._calculate_statistics(tracking_results)
        
        return tracking_results
    
    def _detect_people(self, frame: np.ndarray) -> np.ndarray:
        """Detect people in frame"""
        if self.detector is None:
            return np.empty((0, 5))
        
        try:
            # Use lower confidence threshold and NMS
            conf_threshold = self.config['detection'].get('conf_threshold', 0.15)
            iou_threshold = self.config['detection'].get('iou_threshold', 0.45)
            
            results = self.detector(
                frame, 
                conf=conf_threshold, 
                iou=iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        detections.append([x1, y1, x2, y2, conf])
            
            return np.array(detections) if detections else np.empty((0, 5))
        except Exception as e:
            print(f"Detection error: {e}")
            return np.empty((0, 5))
    
    def _visualize_tracks(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Draw bounding boxes and track IDs"""
        # Generate consistent colors for each ID
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(1000, 3))
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get color for this ID
            color = colors[int(track_id) % 1000].tolist()
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            label = f'ID: {int(track_id)}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate tracking statistics"""
        # Extract all unique track IDs
        all_track_ids = set()
        for frame_data in results['tracks']:
            for detection in frame_data['detections']:
                all_track_ids.add(detection['track_id'])
        
        # Calculate track lengths
        track_lengths = {tid: 0 for tid in all_track_ids}
        for frame_data in results['tracks']:
            for detection in frame_data['detections']:
                track_lengths[detection['track_id']] += 1
        
        if len(track_lengths) == 0:
            return {
                'total_tracks': 0,
                'avg_track_length': 0,
                'max_track_length': 0,
                'min_track_length': 0
            }
        
        return {
            'total_tracks': len(all_track_ids),
            'avg_track_length': float(np.mean(list(track_lengths.values()))),
            'max_track_length': int(max(track_lengths.values())),
            'min_track_length': int(min(track_lengths.values()))
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save tracking results to JSON"""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


# Configuration
default_config = {
    'detection': {
        'conf_threshold': 0.25
    },
    'tracking': {
        'max_dist': 0.2,
        'max_iou_distance': 0.7,
        'max_age': 30,
        'n_init': 3
    }
}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSORT Video Tracker')
    parser.add_argument('--input', type=str, required=True, help='Input video or image sequence')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--detection-model', type=str, default='models/checkpoints/yolov8n.pt',
                       help='Path to detection model')
    parser.add_argument('--reid-model', type=str, help='Path to ReID model (optional)')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = default_config
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Initialize tracker
    tracker = DeepSORTVideoTracker(
        detection_model_path=args.detection_model,
        reid_model_path=args.reid_model,
        config=config
    )
    
    # Process video
    results = tracker.process_video(
        video_path=args.input,
        output_path=args.output,
        visualize=True
    )
    
    # Save results
    if args.output:
        results_path = Path(args.output).with_suffix('.json')
        tracker.save_results(results, str(results_path))
    
    print(f"\nStatistics: {results['statistics']}")

