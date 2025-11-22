"""Crowd density estimation from tracking results"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Tuple, Optional
from collections import deque

# Optional dependency
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some features may be limited.")


class CrowdDensityEstimator:
    """
    Estimate crowd density from tracking results
    Generates density maps and statistics
    """
    def __init__(self, frame_shape: Tuple[int, int], grid_size: int = 50):
        """
        Args:
            frame_shape: (height, width) of frames
            grid_size: Size of grid cells for grid-based density
        """
        self.frame_shape = frame_shape
        self.grid_size = grid_size
        self.grid_h = frame_shape[0] // grid_size
        self.grid_w = frame_shape[1] // grid_size
        
        # Historical data
        self.density_history = deque(maxlen=100)
    
    def compute_density_map(self, tracks: List, method: str = 'gaussian') -> np.ndarray:
        """
        Compute density map from current tracks
        
        Args:
            tracks: List of [x1, y1, x2, y2, track_id, ...] or similar format
            method: 'gaussian' or 'grid'
        
        Returns:
            Density map as numpy array (same shape as frame)
        """
        if method == 'gaussian':
            return self._gaussian_density(tracks)
        elif method == 'grid':
            return self._grid_density(tracks)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gaussian' or 'grid'")
    
    def _gaussian_density(self, tracks: List) -> np.ndarray:
        """Create Gaussian-based density map"""
        density_map = np.zeros(self.frame_shape[:2], dtype=np.float32)
        
        for track in tracks:
            if len(track) < 4:
                continue
            
            x1, y1, x2, y2 = track[:4]
            
            # Get center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            if 0 <= cx < self.frame_shape[1] and 0 <= cy < self.frame_shape[0]:
                density_map[cy, cx] += 1
        
        # Apply Gaussian blur
        density_map = gaussian_filter(density_map, sigma=20)
        
        return density_map
    
    def _grid_density(self, tracks: List) -> np.ndarray:
        """Create grid-based density map"""
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        
        for track in tracks:
            if len(track) < 4:
                continue
            
            x1, y1, x2, y2 = track[:4]
            
            # Get center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Map to grid
            grid_x = min(cx // self.grid_size, self.grid_w - 1)
            grid_y = min(cy // self.grid_size, self.grid_h - 1)
            
            if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h:
                grid[grid_y, grid_x] += 1
        
        # Resize to original frame size
        density_map = cv2.resize(grid.astype(np.float32), 
                                (self.frame_shape[1], self.frame_shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
        
        return density_map
    
    def visualize_density(self, density_map: np.ndarray, frame: Optional[np.ndarray] = None, 
                         alpha: float = 0.6) -> np.ndarray:
        """
        Visualize density map as heatmap overlay
        
        Args:
            density_map: Density map from compute_density_map
            frame: Optional background frame
            alpha: Transparency for overlay
        
        Returns:
            Visualization image
        """
        # Normalize density map
        if density_map.max() > 0:
            density_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
        else:
            density_norm = density_map
        density_norm = (density_norm * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        
        if frame is not None:
            # Overlay on frame
            vis = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        else:
            vis = heatmap
        
        return vis
    
    def compute_statistics(self, tracks: List) -> Dict:
        """
        Compute crowd statistics
        
        Args:
            tracks: List of tracks
        
        Returns:
            Dictionary with statistics
        """
        if len(tracks) == 0:
            return {
                'count': 0,
                'density': 0.0,
                'distribution': 'uniform',
                'hotspots': []
            }
        
        # Count
        count = len(tracks)
        
        # Density (people per unit area)
        frame_area = self.frame_shape[0] * self.frame_shape[1]
        density = count / frame_area * 1000000  # per megapixel
        
        # Compute density map for distribution analysis
        density_map = self.compute_density_map(tracks, method='grid')
        
        # Find hotspots (high density regions)
        if density_map.max() > 0:
            threshold = np.percentile(density_map, 90)
            hotspot_mask = density_map > threshold
            
            # Get hotspot locations
            hotspot_coords = np.argwhere(hotspot_mask)
            hotspots = []
            
            if len(hotspot_coords) > 0 and SKLEARN_AVAILABLE:
                try:
                    # Cluster hotspot coordinates
                    clustering = DBSCAN(eps=50, min_samples=5).fit(hotspot_coords)
                    
                    for label in set(clustering.labels_):
                        if label == -1:
                            continue
                        
                        cluster_points = hotspot_coords[clustering.labels_ == label]
                        center = cluster_points.mean(axis=0)
                        
                        hotspots.append({
                            'location': center.tolist(),
                            'count': len(cluster_points)
                        })
                except Exception:
                    # Fallback if clustering fails
                    hotspots = [{'location': hotspot_coords.mean(axis=0).tolist(), 
                                'count': len(hotspot_coords)}]
            else:
                # Simple hotspot detection without clustering
                if len(hotspot_coords) > 0:
                    hotspots = [{'location': hotspot_coords.mean(axis=0).tolist(), 
                                'count': len(hotspot_coords)}]
        else:
            hotspots = []
        
        # Distribution uniformity
        std_density = np.std(density_map)
        mean_density = np.mean(density_map)
        uniformity = 1 - (std_density / (mean_density + 1e-8))
        
        if uniformity > 0.7:
            distribution = 'uniform'
        elif uniformity > 0.4:
            distribution = 'clustered'
        else:
            distribution = 'highly_clustered'
        
        return {
            'count': count,
            'density': float(density),
            'distribution': distribution,
            'uniformity': float(uniformity),
            'hotspots': hotspots
        }
    
    def update_history(self, tracks: List):
        """Update historical density data"""
        stats = self.compute_statistics(tracks)
        self.density_history.append(stats)
    
    def get_temporal_statistics(self) -> Dict:
        """Get statistics over time"""
        if len(self.density_history) == 0:
            return {}
        
        counts = [s['count'] for s in self.density_history]
        densities = [s['density'] for s in self.density_history]
        
        return {
            'avg_count': float(np.mean(counts)),
            'max_count': int(np.max(counts)),
            'min_count': int(np.min(counts)),
            'avg_density': float(np.mean(densities)),
            'count_variance': float(np.var(counts)),
            'trend': self._compute_trend(counts)
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend (increasing, decreasing, stable)"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
        except:
            return 'stable'
        
        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        else:
            return 'stable'


