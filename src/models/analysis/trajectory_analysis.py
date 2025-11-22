"""Trajectory analysis and prediction"""
import numpy as np
import cv2
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Optional dependencies
try:
    from sklearn.cluster import DBSCAN, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Clustering features may be limited.")


class TrajectoryAnalyzer:
    """
    Analyze and predict trajectories
    Detect anomalies and unusual patterns
    """
    def __init__(self, frame_shape: Tuple[int, int]):
        """
        Args:
            frame_shape: (height, width) of frames
        """
        self.frame_shape = frame_shape
        self.trajectories = defaultdict(list)  # {track_id: [(x, y, t), ...]}
        self.max_trajectory_length = 300
    
    def update(self, tracks: List, frame_id: int):
        """Update trajectories with new tracks"""
        for track in tracks:
            if len(track) < 5:
                continue
            
            x1, y1, x2, y2, track_id = track[:5]
            
            # Get center point
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Add to trajectory
            self.trajectories[track_id].append((cx, cy, frame_id))
            
            # Limit length
            if len(self.trajectories[track_id]) > self.max_trajectory_length:
                self.trajectories[track_id].pop(0)
    
    def compute_velocity(self, track_id: int, window: int = 5) -> Tuple[float, float, float, float]:
        """
        Compute velocity for a track
        
        Returns:
            (vx, vy, speed, direction) where direction is in degrees
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        
        traj = self.trajectories[track_id][-window:]
        
        if len(traj) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        
        # Compute velocity
        x1, y1, t1 = traj[0]
        x2, y2, t2 = traj[-1]
        
        dt = max(t2 - t1, 1)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        # Speed and direction
        speed = np.sqrt(vx**2 + vy**2)
        direction = np.arctan2(vy, vx) * 180 / np.pi  # degrees
        
        return (vx, vy, speed, direction)
    
    def predict_position(self, track_id: int, n_frames: int = 10, method: str = 'linear') -> List[Tuple[float, float]]:
        """
        Predict future position
        
        Args:
            track_id: Track to predict
            n_frames: Number of frames to predict ahead
            method: 'linear' or 'kalman'
        
        Returns:
            List of predicted (x, y) positions
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 5:
            return []
        
        if method == 'linear':
            return self._linear_prediction(track_id, n_frames)
        elif method == 'kalman':
            return self._kalman_prediction(track_id, n_frames)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear' or 'kalman'")
    
    def _linear_prediction(self, track_id: int, n_frames: int) -> List[Tuple[float, float]]:
        """Simple linear extrapolation"""
        vx, vy, _, _ = self.compute_velocity(track_id)
        
        # Current position
        x, y, _ = self.trajectories[track_id][-1]
        
        # Predict
        predictions = []
        for i in range(1, n_frames + 1):
            pred_x = x + vx * i
            pred_y = y + vy * i
            predictions.append((pred_x, pred_y))
        
        return predictions
    
    def _kalman_prediction(self, track_id: int, n_frames: int) -> List[Tuple[float, float]]:
        """Kalman filter-based prediction"""
        # Use last velocity
        vx, vy, _, _ = self.compute_velocity(track_id)
        
        # Current position
        x, y, _ = self.trajectories[track_id][-1]
        
        # Predict with noise consideration
        predictions = []
        noise_std = 2.0
        
        for i in range(1, n_frames + 1):
            pred_x = x + vx * i + np.random.normal(0, noise_std)
            pred_y = y + vy * i + np.random.normal(0, noise_std)
            predictions.append((pred_x, pred_y))
        
        return predictions
    
    def detect_anomalies(self, track_id: int) -> List[str]:
        """
        Detect anomalous behavior in trajectory
        
        Returns:
            List of anomaly types detected
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 10:
            return []
        
        anomalies = []
        
        # Check for sudden stops
        velocities = []
        for i in range(5, len(self.trajectories[track_id])):
            vx, vy, speed, _ = self.compute_velocity(track_id, window=5)
            velocities.append(speed)
        
        if len(velocities) > 0:
            # Sudden stop
            if len(velocities) >= 15:
                recent_speed = np.mean(velocities[-5:])
                previous_speed = np.mean(velocities[-15:-5])
                if recent_speed < 1.0 and previous_speed > 5.0:
                    anomalies.append('sudden_stop')
            
            # Erratic movement
            if np.std(velocities) > 10.0:
                anomalies.append('erratic_movement')
        
        # Check for loitering (staying in small area)
        recent_traj = np.array(self.trajectories[track_id][-30:])
        if len(recent_traj) >= 30:
            positions = recent_traj[:, :2]
            spread = np.std(positions, axis=0).sum()
            if spread < 20:
                anomalies.append('loitering')
        
        # Check for wrong-way movement (if flow direction is known)
        _, _, _, direction = self.compute_velocity(track_id)
        # Assume normal flow is left-to-right (0 to 90 degrees)
        if -180 < direction < -90 or 90 < direction < 180:
            anomalies.append('wrong_way')
        
        return anomalies
    
    def cluster_trajectories(self, method: str = 'dbscan') -> Dict[int, List[int]]:
        """
        Cluster trajectories to find common paths
        
        Returns:
            Dictionary mapping cluster_id to list of track_ids
        """
        if len(self.trajectories) < 2:
            return {}
        
        # Extract trajectory features (start, end, direction, length)
        features = []
        track_ids = []
        
        for track_id, traj in self.trajectories.items():
            if len(traj) < 10:
                continue
            
            # Start and end positions
            start_x, start_y, _ = traj[0]
            end_x, end_y, _ = traj[-1]
            
            # Direction
            _, _, _, direction = self.compute_velocity(track_id)
            
            # Length
            length = len(traj)
            
            features.append([start_x, start_y, end_x, end_y, direction, length])
            track_ids.append(track_id)
        
        if len(features) == 0:
            return {}
        
        features = np.array(features)
        
        # Normalize features
        features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Cluster
        if not SKLEARN_AVAILABLE:
            # Simple clustering fallback
            return {}
        
        try:
            if method == 'dbscan':
                clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_norm)
            else:
                clustering = KMeans(n_clusters=min(5, len(features))).fit(features_norm)
            
            # Group by cluster
            clusters = defaultdict(list)
            for track_id, label in zip(track_ids, clustering.labels_):
                if label != -1:  # Ignore noise
                    clusters[int(label)].append(track_id)
            
            return clusters
        except Exception as e:
            print(f"Warning: Clustering failed: {e}")
            return {}
    
    def visualize_trajectories(self, background: Optional[np.ndarray] = None, 
                              track_ids: Optional[List[int]] = None, 
                              show_predictions: bool = False) -> np.ndarray:
        """
        Visualize trajectories
        
        Args:
            background: Background frame
            track_ids: Specific tracks to show (None = all)
            show_predictions: Whether to show predicted positions
        
        Returns:
            Visualization image
        """
        if background is None:
            canvas = np.ones((self.frame_shape[0], self.frame_shape[1], 3), dtype=np.uint8) * 255
        else:
            canvas = background.copy()
        
        # Generate colors
        np.random.seed(42)
        colors = np.random.randint(0, 255, (1000, 3))
        
        # Draw trajectories
        track_list = track_ids if track_ids else list(self.trajectories.keys())
        
        for track_id in track_list:
            if track_id not in self.trajectories:
                continue
            
            traj = self.trajectories[track_id]
            color = colors[int(track_id) % 1000].tolist()
            
            # Draw path
            for i in range(1, len(traj)):
                x1, y1, _ = traj[i-1]
                x2, y2, _ = traj[i]
                
                cv2.line(canvas, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color, 2)
            
            # Draw current position
            if len(traj) > 0:
                x, y, _ = traj[-1]
                cv2.circle(canvas, (int(x), int(y)), 5, color, -1)
                cv2.putText(canvas, str(track_id), (int(x)+10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw predictions
            if show_predictions:
                predictions = self.predict_position(track_id, n_frames=30)
                for px, py in predictions:
                    cv2.circle(canvas, (int(px), int(py)), 2, color, -1)
        
        return canvas
    
    def generate_flow_map(self) -> np.ndarray:
        """
        Generate flow visualization showing dominant movement directions
        
        Returns:
            Flow map image with arrows
        """
        canvas = np.ones((self.frame_shape[0], self.frame_shape[1], 3), dtype=np.uint8) * 255
        
        # Create grid
        grid_size = 50
        grid_h = self.frame_shape[0] // grid_size
        grid_w = self.frame_shape[1] // grid_size
        
        # Accumulate velocities in each grid cell
        grid_vx = np.zeros((grid_h, grid_w))
        grid_vy = np.zeros((grid_h, grid_w))
        grid_count = np.zeros((grid_h, grid_w))
        
        for track_id in self.trajectories:
            vx, vy, _, _ = self.compute_velocity(track_id)
            
            # Get recent positions
            recent = self.trajectories[track_id][-10:]
            for x, y, _ in recent:
                gx = min(int(x // grid_size), grid_w - 1)
                gy = min(int(y // grid_size), grid_h - 1)
                
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    grid_vx[gy, gx] += vx
                    grid_vy[gy, gx] += vy
                    grid_count[gy, gx] += 1
        
        # Average velocities
        mask = grid_count > 0
        grid_vx[mask] /= grid_count[mask]
        grid_vy[mask] /= grid_count[mask]
        
        # Draw arrows
        for gy in range(grid_h):
            for gx in range(grid_w):
                if grid_count[gy, gx] < 2:
                    continue
                
                cx = int((gx + 0.5) * grid_size)
                cy = int((gy + 0.5) * grid_size)
                
                vx = grid_vx[gy, gx] * 5  # Scale for visibility
                vy = grid_vy[gy, gx] * 5
                
                # Draw arrow
                end_x = int(cx + vx)
                end_y = int(cy + vy)
                
                cv2.arrowedLine(canvas,
                              (cx, cy),
                              (end_x, end_y),
                              (0, 0, 255), 2, tipLength=0.3)
        
        return canvas


