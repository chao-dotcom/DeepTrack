"""YOLOv8 Person Detector"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class YOLODetector:
    """YOLOv8-based person detector"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLOv8 model. If None, uses 'yolov8n.pt' (auto-downloads)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            if self.model_path is None:
                # Use default model (will auto-download)
                self.model_path = 'yolov8n.pt'
            
            print(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✓ Model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install it with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect people in a frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detections, each with:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - class_id: Class ID (0 for person in COCO)
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for person class (class 0 in COCO)
                if int(box.cls) == 0 and float(box.conf) >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(box.conf),
                        'class_id': 0,
                        'class_name': 'person'
                    })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections from detect()
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_copy


