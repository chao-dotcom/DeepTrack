"""Multi-camera tracking system"""
import numpy as np
import torch
import cv2
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import DeepSORT tracker
try:
    from src.models.tracking.deepsort import DeepSORT
except ImportError:
    # Fallback if DeepSORT not available
    DeepSORT = None


class MultiCameraTracker:
    """
    Track people across multiple cameras
    Handles camera handoff and global ID management
    """
    def __init__(self, num_cameras: int, reid_model=None, homography_matrices: Optional[Dict] = None):
        """
        Args:
            num_cameras: Number of cameras in the system
            reid_model: ReID model for cross-camera matching
            homography_matrices: Dict mapping camera_id to homography matrix for world projection
        """
        self.num_cameras = num_cameras
        self.reid_model = reid_model
        self.homography_matrices = homography_matrices or {}
        
        # Per-camera trackers
        if DeepSORT is not None and reid_model is not None:
            self.camera_trackers = [
                DeepSORT(reid_model) for _ in range(num_cameras)
            ]
        else:
            self.camera_trackers = [None] * num_cameras
            print("Warning: DeepSORT or ReID model not available. Multi-camera tracking may be limited.")
        
        # Global ID management
        self.global_id_counter = 1
        self.local_to_global = defaultdict(dict)  # {camera_id: {local_id: global_id}}
        self.global_features = defaultdict(list)  # {global_id: [features]}
        
        # Camera topology (which cameras can see the same space)
        self.camera_neighbors = self._build_camera_graph()
        
        # Device for ReID model
        if reid_model is not None:
            self.device = next(reid_model.parameters()).device
        else:
            self.device = torch.device('cpu')
    
    def _build_camera_graph(self):
        """Define which cameras have overlapping fields of view"""
        # Default: assume sequential cameras overlap
        # Override this method for custom camera topology
        neighbors = defaultdict(list)
        for i in range(self.num_cameras - 1):
            neighbors[i].append(i + 1)
            neighbors[i + 1].append(i)
        return neighbors
    
    def set_camera_topology(self, topology: Dict[int, List[int]]):
        """
        Set custom camera topology
        
        Args:
            topology: Dict mapping camera_id to list of neighboring camera IDs
        """
        self.camera_neighbors = defaultdict(list, topology)
    
    def update(self, camera_id: int, detections: np.ndarray, frame: np.ndarray) -> List:
        """
        Update tracker for a specific camera
        
        Args:
            camera_id: Camera identifier (0 to num_cameras-1)
            detections: Detections from this camera [N, 5] (x1, y1, x2, y2, conf)
            frame: Current frame from this camera
        
        Returns:
            List of tracks with global IDs [x1, y1, x2, y2, global_id, camera_id]
        """
        if camera_id < 0 or camera_id >= self.num_cameras:
            raise ValueError(f"Invalid camera_id: {camera_id}. Must be 0-{self.num_cameras-1}")
        
        # Get local tracks from this camera's tracker
        if self.camera_trackers[camera_id] is not None:
            local_tracks = self.camera_trackers[camera_id].update(detections, frame)
        else:
            # Fallback: simple tracking without DeepSORT
            local_tracks = self._simple_track(detections)
        
        # Extract features for cross-camera matching
        track_features = []
        for track in local_tracks:
            x1, y1, x2, y2, local_id = track[:5]
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            feature = self._extract_reid_feature(crop)
            track_features.append((local_id, feature))
        
        # Assign global IDs
        global_tracks = []
        for local_id, feature in track_features:
            global_id = self._assign_global_id(camera_id, local_id, feature)
            
            # Store feature for this global ID
            self.global_features[global_id].append(feature)
            if len(self.global_features[global_id]) > 100:
                self.global_features[global_id].pop(0)
            
            # Find corresponding track
            for track in local_tracks:
                if len(track) >= 5 and track[4] == local_id:
                    global_tracks.append([*track[:4], global_id, camera_id])
                    break
        
        return global_tracks
    
    def _simple_track(self, detections: np.ndarray) -> List:
        """Simple tracking fallback when DeepSORT unavailable"""
        tracks = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf = det[:5]
            tracks.append([x1, y1, x2, y2, i + 1])  # Simple ID assignment
        return tracks
    
    def _assign_global_id(self, camera_id: int, local_id: int, feature: np.ndarray) -> int:
        """
        Assign global ID to a local track
        Handles new tracks and cross-camera re-identification
        """
        # Check if this local ID already has a global ID
        if local_id in self.local_to_global[camera_id]:
            return self.local_to_global[camera_id][local_id]
        
        # Try to match with existing global IDs from neighboring cameras
        best_match_id = None
        best_similarity = 0.0
        similarity_threshold = 0.7
        
        for neighbor_cam in self.camera_neighbors[camera_id]:
            for local_id_neighbor, global_id in self.local_to_global[neighbor_cam].items():
                # Get features for this global ID
                if global_id in self.global_features and len(self.global_features[global_id]) > 0:
                    global_feats = np.array(self.global_features[global_id])
                    
                    # Compute similarity (cosine similarity)
                    similarities = np.dot(global_feats, feature)
                    max_similarity = np.max(similarities)
                    
                    if max_similarity > best_similarity and max_similarity > similarity_threshold:
                        best_similarity = max_similarity
                        best_match_id = global_id
        
        # If good match found, use that global ID
        if best_match_id is not None:
            self.local_to_global[camera_id][local_id] = best_match_id
            return best_match_id
        
        # Otherwise, create new global ID
        new_global_id = self.global_id_counter
        self.global_id_counter += 1
        self.local_to_global[camera_id][local_id] = new_global_id
        
        return new_global_id
    
    def _extract_reid_feature(self, crop: np.ndarray) -> np.ndarray:
        """Extract ReID feature from crop"""
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            # Return zero feature if crop is invalid
            return np.zeros(2048, dtype=np.float32)
        
        if self.reid_model is None:
            # Return random feature if no ReID model
            return np.random.randn(2048).astype(np.float32)
        
        try:
            # Transform and extract feature
            from torchvision import transforms
            from PIL import Image
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Convert BGR to RGB if needed
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = crop
            
            crop_tensor = transform(crop_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.reid_model(crop_tensor, return_features=True)
            
            return feature.cpu().numpy()[0]
        except Exception as e:
            print(f"Warning: ReID feature extraction failed: {e}")
            return np.zeros(2048, dtype=np.float32)
    
    def project_to_world_coordinates(self, camera_id: int, bbox: List[float]) -> np.ndarray:
        """
        Project bounding box to world coordinates using homography
        Useful for top-down view visualization
        
        Args:
            camera_id: Camera identifier
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            World coordinates [x, y] of foot position
        """
        if camera_id not in self.homography_matrices:
            # Return center if no homography
            return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        H = self.homography_matrices[camera_id]
        
        # Get bottom center of bbox (foot position)
        x = (bbox[0] + bbox[2]) / 2
        y = bbox[3]
        
        # Apply homography
        point = np.array([x, y, 1.0], dtype=np.float32)
        world_point = H @ point
        world_point = world_point[:2] / world_point[2]
        
        return world_point
    
    def visualize_top_down(self, global_tracks: List, frame_size: Tuple[int, int] = (1000, 1000)) -> np.ndarray:
        """
        Create top-down view of all tracks across cameras
        
        Args:
            global_tracks: List of tracks [x1, y1, x2, y2, global_id, camera_id]
            frame_size: Size of output canvas (width, height)
        
        Returns:
            Visualization image
        """
        canvas = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
        
        # Draw camera FOVs (simplified representation)
        for camera_id in range(self.num_cameras):
            # Draw camera position (simplified)
            cam_x = int((camera_id + 1) * frame_size[0] / (self.num_cameras + 1))
            cam_y = frame_size[1] - 50
            cv2.circle(canvas, (cam_x, cam_y), 20, (100, 100, 100), -1)
            cv2.putText(canvas, f'Cam{camera_id}', (cam_x - 30, cam_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw tracks
        np.random.seed(42)
        colors = np.random.randint(0, 255, (1000, 3))
        
        for track in global_tracks:
            if len(track) < 6:
                continue
            
            x1, y1, x2, y2, global_id, camera_id = track[:6]
            
            # Project to world coordinates
            world_pos = self.project_to_world_coordinates(camera_id, [x1, y1, x2, y2])
            
            # Scale to canvas size (assuming world coordinates are in pixel space)
            # Adjust scaling based on your actual world coordinate system
            canvas_x = int(world_pos[0] * frame_size[0] / 1920)  # Assuming 1920 is max width
            canvas_y = int(world_pos[1] * frame_size[1] / 1080)  # Assuming 1080 is max height
            
            # Clamp to canvas bounds
            canvas_x = max(0, min(canvas_x, frame_size[0] - 1))
            canvas_y = max(0, min(canvas_y, frame_size[1] - 1))
            
            # Draw
            color = colors[int(global_id) % 1000].tolist()
            cv2.circle(canvas, (canvas_x, canvas_y), 10, color, -1)
            cv2.putText(canvas, f'ID:{int(global_id)}', (canvas_x + 15, canvas_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return canvas
    
    def get_global_statistics(self) -> Dict:
        """Get statistics about global tracking"""
        total_global_ids = len(self.global_features)
        camera_assignments = {cam_id: len(local_dict) for cam_id, local_dict in self.local_to_global.items()}
        
        return {
            'total_global_ids': total_global_ids,
            'camera_assignments': camera_assignments,
            'cross_camera_matches': self._count_cross_camera_matches()
        }
    
    def _count_cross_camera_matches(self) -> int:
        """Count how many global IDs appear in multiple cameras"""
        global_id_cameras = defaultdict(set)
        
        for cam_id, local_dict in self.local_to_global.items():
            for local_id, global_id in local_dict.items():
                global_id_cameras[global_id].add(cam_id)
        
        # Count IDs that appear in multiple cameras
        cross_camera_count = sum(1 for cameras in global_id_cameras.values() if len(cameras) > 1)
        
        return cross_camera_count


