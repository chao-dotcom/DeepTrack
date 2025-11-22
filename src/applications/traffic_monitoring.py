"""Traffic flow and violation detection"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import time


class TrafficMonitor:
    """
    Monitor traffic flow and detect violations
    """
    def __init__(self, tracker, flow_direction: Optional[float] = None):
        """
        Args:
            tracker: Tracking system instance
            flow_direction: Expected flow direction in degrees (0 = right, 90 = down)
        """
        self.tracker = tracker
        self.flow_direction = flow_direction
        self.violations = []
        self.flow_statistics = defaultdict(int)
        self.start_time = time.time()
        
        # Track trajectories for analysis
        self.track_histories = defaultdict(list)
    
    def detect_violations(self, tracks: List, frame_id: int) -> List[Dict]:
        """
        Detect traffic violations
        
        Args:
            tracks: List of tracks
            frame_id: Current frame number
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        for track in tracks:
            if len(track) < 5:
                continue
            
            track_id = track[4]
            bbox = track[:4]
            
            # Update track history
            self.track_histories[track_id].append({
                'frame': frame_id,
                'bbox': bbox,
                'time': time.time()
            })
            
            # Limit history
            if len(self.track_histories[track_id]) > 100:
                self.track_histories[track_id].pop(0)
            
            # Wrong-way detection
            if self._is_wrong_way(track_id):
                violations.append({
                    'type': 'wrong_way',
                    'track_id': track_id,
                    'frame': frame_id,
                    'bbox': bbox
                })
            
            # Jaywalking detection
            if self._is_jaywalking(track_id):
                violations.append({
                    'type': 'jaywalking',
                    'track_id': track_id,
                    'frame': frame_id,
                    'bbox': bbox
                })
            
            # Loitering detection
            if self._is_loitering(track_id):
                violations.append({
                    'type': 'loitering',
                    'track_id': track_id,
                    'frame': frame_id,
                    'bbox': bbox
                })
        
        self.violations.extend(violations)
        return violations
    
    def _is_wrong_way(self, track_id: int) -> bool:
        """Detect wrong-way movement"""
        if track_id not in self.track_histories or len(self.track_histories[track_id]) < 10:
            return False
        
        if self.flow_direction is None:
            return False
        
        history = self.track_histories[track_id]
        
        # Compute average direction
        directions = []
        for i in range(5, len(history)):
            prev = history[i-5]
            curr = history[i]
            
            dx = curr['bbox'][0] - prev['bbox'][0]
            dy = curr['bbox'][1] - prev['bbox'][1]
            
            direction = np.arctan2(dy, dx) * 180 / np.pi
            
            # Normalize to 0-360
            if direction < 0:
                direction += 360
            
            directions.append(direction)
        
        if len(directions) == 0:
            return False
        
        avg_direction = np.mean(directions)
        
        # Check if direction is opposite to flow (within 180 degrees)
        direction_diff = abs(avg_direction - self.flow_direction)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff
        
        # Wrong way if more than 90 degrees from expected flow
        return direction_diff > 90
    
    def _is_jaywalking(self, track_id: int) -> bool:
        """Detect jaywalking (crossing outside designated areas)"""
        # This is a simplified version
        # In practice, you'd check against crosswalk polygons
        
        if track_id not in self.track_histories or len(self.track_histories[track_id]) < 20:
            return False
        
        history = self.track_histories[track_id]
        
        # Check for rapid lateral movement (crossing road)
        lateral_movements = []
        for i in range(10, len(history)):
            prev = history[i-10]
            curr = history[i]
            
            # Assuming y-axis is vertical (road direction)
            lateral_movement = abs(curr['bbox'][0] - prev['bbox'][0])
            lateral_movements.append(lateral_movement)
        
        if len(lateral_movements) == 0:
            return False
        
        # Jaywalking if significant lateral movement
        avg_lateral = np.mean(lateral_movements)
        return avg_lateral > 50  # Threshold in pixels
    
    def _is_loitering(self, track_id: int, threshold_seconds: float = 30.0) -> bool:
        """Detect loitering (staying in one place too long)"""
        if track_id not in self.track_histories or len(self.track_histories[track_id]) < 10:
            return False
        
        history = self.track_histories[track_id]
        
        # Check time span
        time_span = history[-1]['time'] - history[0]['time']
        
        if time_span < threshold_seconds:
            return False
        
        # Check spatial spread
        positions = [h['bbox'][:2] for h in history]
        positions = np.array(positions)
        
        spread = np.std(positions, axis=0).sum()
        
        # Loitering if low spatial spread over long time
        return spread < 30
    
    def compute_flow_metrics(self) -> Dict:
        """Compute traffic flow metrics"""
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        
        return {
            'total_count': sum(self.flow_statistics.values()),
            'flow_by_direction': dict(self.flow_statistics),
            'peak_hour': self._identify_peak_hours(),
            'violations_per_hour': len(self.violations) / max(1, elapsed_hours),
            'total_violations': len(self.violations)
        }
    
    def _identify_peak_hours(self) -> List[int]:
        """Identify peak traffic hours (placeholder)"""
        # In practice, you'd analyze traffic over time
        # For now, return empty list
        return []
    
    def get_violation_summary(self) -> Dict:
        """Get summary of violations"""
        violation_types = defaultdict(int)
        
        for violation in self.violations:
            violation_types[violation['type']] += 1
        
        return {
            'total': len(self.violations),
            'by_type': dict(violation_types),
            'recent': self.violations[-10:] if len(self.violations) > 0 else []
        }


