"""Retail store people counting and analytics"""
import numpy as np
import cv2
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class RetailAnalytics:
    """
    Analyze customer behavior in retail environment
    """
    def __init__(self, tracker, zone_definitions: Dict[str, List[Tuple[int, int]]]):
        """
        Args:
            tracker: Tracking system instance
            zone_definitions: Dict mapping zone_name to list of polygon vertices
        """
        self.tracker = tracker
        self.zones = zone_definitions
        self.zone_statistics = defaultdict(lambda: {
            'entries': 0,
            'exits': 0,
            'current_count': 0,
            'dwell_times': [],
            'trajectories': []
        })
        
        # Track zone history per person
        self.track_zones = defaultdict(dict)
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def update(self, frame: np.ndarray, frame_id: int):
        """
        Process frame and update analytics
        
        Args:
            frame: Current frame
            frame_id: Frame number
        
        Returns:
            (tracks, insights) tuple
        """
        # Get tracks from tracker
        if hasattr(self.tracker, 'update'):
            # Assume tracker.update returns detections/tracks
            detections = self.tracker.detect(frame) if hasattr(self.tracker, 'detect') else []
            tracks = self.tracker.update(detections, frame) if hasattr(self.tracker, 'update') else []
        else:
            tracks = []
        
        # Analyze each track
        for track in tracks:
            if len(track) < 5:
                continue
            
            track_id = track[4]
            bbox = track[:4]
            
            # Check zone transitions
            self._check_zone_transitions(track_id, bbox)
            
            # Update dwell time
            self._update_dwell_time(track_id, frame_id)
        
        # Generate insights
        insights = self.generate_insights()
        
        return tracks, insights
    
    def _check_zone_transitions(self, track_id: int, bbox: List[float]):
        """Check if track entered/exited any zones"""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        for zone_name, polygon in self.zones.items():
            is_inside = self._point_in_polygon(center, polygon)
            
            was_inside = self.track_zones[track_id].get(zone_name, False)
            
            if is_inside and not was_inside:
                # Entry event
                self.zone_statistics[zone_name]['entries'] += 1
                self.zone_statistics[zone_name]['current_count'] += 1
                self.track_zones[track_id][zone_name] = True
                self.track_zones[track_id][f'{zone_name}_entry_time'] = time.time()
                
            elif not is_inside and was_inside:
                # Exit event
                self.zone_statistics[zone_name]['exits'] += 1
                self.zone_statistics[zone_name]['current_count'] = max(0, 
                    self.zone_statistics[zone_name]['current_count'] - 1)
                self.track_zones[track_id][zone_name] = False
                
                # Calculate dwell time
                entry_time = self.track_zones[track_id].get(f'{zone_name}_entry_time')
                if entry_time:
                    dwell_time = time.time() - entry_time
                    self.zone_statistics[zone_name]['dwell_times'].append(dwell_time)
    
    def _update_dwell_time(self, track_id: int, frame_id: int):
        """Update dwell time tracking"""
        # This is handled in _check_zone_transitions
        pass
    
    def generate_insights(self) -> Dict:
        """Generate actionable insights"""
        insights = {}
        
        for zone_name, stats in self.zone_statistics.items():
            avg_dwell = np.mean(stats['dwell_times']) if stats['dwell_times'] else 0
            
            insights[zone_name] = {
                'current_occupancy': stats['current_count'],
                'total_visitors': stats['entries'],
                'avg_dwell_time_seconds': float(avg_dwell),
                'popularity_score': stats['entries'] * avg_dwell,
                'conversion_rate': self._compute_conversion_rate(zone_name)
            }
        
        # Identify hot zones
        zone_scores = [(zone, data.get('popularity_score', 0)) 
                      for zone, data in insights.items() 
                      if isinstance(data, dict)]
        
        if zone_scores:
            hot_zones = sorted(zone_scores, key=lambda x: x[1], reverse=True)[:3]
            insights['hot_zones'] = [z[0] for z in hot_zones]
        else:
            insights['hot_zones'] = []
        
        # Identify cold zones (low traffic)
        cold_zones = [
            zone for zone, data in insights.items()
            if isinstance(data, dict) and data.get('total_visitors', 0) < 5
        ]
        
        insights['cold_zones'] = cold_zones
        
        return insights
    
    def _compute_conversion_rate(self, zone_name: str) -> float:
        """Compute conversion rate (placeholder - implement based on business logic)"""
        # This would typically compare visitors to purchases
        # For now, return a placeholder
        entries = self.zone_statistics[zone_name]['entries']
        if entries == 0:
            return 0.0
        
        # Placeholder: assume 10% conversion
        return 0.1
    
    def visualize_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Create occupancy heatmap overlay"""
        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        for zone_name, polygon in self.zones.items():
            count = self.zone_statistics[zone_name]['current_count']
            
            # Fill polygon with intensity based on count
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            polygon_array = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [polygon_array], 1)
            
            heatmap += mask * count * 20  # Scale factor
        
        # Apply colormap
        if heatmap.max() > 0:
            heatmap_norm = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
        else:
            heatmap_norm = heatmap.astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Overlay
        result = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        
        return result


