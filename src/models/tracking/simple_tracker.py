"""Simple Centroid-based Tracker"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class SimpleTracker:
    """Simple centroid-based multi-object tracker"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Maximum frames an object can disappear before removal
            max_distance: Maximum distance for matching (in pixels)
        """
        self.next_id = 0
        self.objects = {}  # {id: {'centroid': (x, y), 'bbox': [x1,y1,x2,y2], 'disappeared': 0}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.track_history = defaultdict(list)  # Track history for visualization
    
    def _calculate_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate centroid from bounding box"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    def _calculate_distance(self, centroid1: Tuple[float, float], 
                           centroid2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centroids"""
        return np.sqrt((centroid1[0] - centroid2[0])**2 + 
                      (centroid1[1] - centroid2[1])**2)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections with 'bbox' key
            
        Returns:
            List of tracked objects with 'id' added
        """
        if len(detections) == 0:
            # No detections - mark all as disappeared
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
            return []
        
        # Calculate centroids for new detections
        input_centroids = []
        for det in detections:
            centroid = self._calculate_centroid(det['bbox'])
            input_centroids.append(centroid)
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for det, centroid in zip(detections, input_centroids):
                self.objects[self.next_id] = {
                    'centroid': centroid,
                    'bbox': det['bbox'],
                    'disappeared': 0
                }
                det['id'] = self.next_id
                self.track_history[self.next_id].append(centroid)
                self.next_id += 1
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]['centroid'] 
                              for obj_id in object_ids]
            
            # Calculate distance matrix
            distances = np.zeros((len(object_ids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, inp_centroid in enumerate(input_centroids):
                    distances[i, j] = self._calculate_distance(obj_centroid, inp_centroid)
            
            # Simple greedy matching
            used_detection_indices = set()
            used_object_indices = set()
            
            # Sort by distance and match
            matches = []
            while True:
                min_dist = np.inf
                min_i, min_j = -1, -1
                
                for i in range(len(object_ids)):
                    if i in used_object_indices:
                        continue
                    for j in range(len(input_centroids)):
                        if j in used_detection_indices:
                            continue
                        if distances[i, j] < min_dist:
                            min_dist = distances[i, j]
                            min_i, min_j = i, j
                
                if min_dist > self.max_distance or min_i == -1:
                    break
                
                matches.append((object_ids[min_i], min_j))
                used_object_indices.add(min_i)
                used_detection_indices.add(min_j)
            
            # Update matched objects
            for obj_id, det_idx in matches:
                det = detections[det_idx]
                centroid = input_centroids[det_idx]
                self.objects[obj_id]['centroid'] = centroid
                self.objects[obj_id]['bbox'] = det['bbox']
                self.objects[obj_id]['disappeared'] = 0
                det['id'] = obj_id
                self.track_history[obj_id].append(centroid)
            
            # Register new objects for unmatched detections
            for j, det in enumerate(detections):
                if j not in used_detection_indices:
                    centroid = input_centroids[j]
                    self.objects[self.next_id] = {
                        'centroid': centroid,
                        'bbox': det['bbox'],
                        'disappeared': 0
                    }
                    det['id'] = self.next_id
                    self.track_history[self.next_id].append(centroid)
                    self.next_id += 1
            
            # Mark unmatched objects as disappeared
            for i, obj_id in enumerate(object_ids):
                if i not in used_object_indices:
                    self.objects[obj_id]['disappeared'] += 1
                    if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                        del self.objects[obj_id]
        
        return detections
    
    def get_track_count(self) -> int:
        """Get current number of active tracks"""
        return len(self.objects)


