"""DeepSORT tracker implementation with Kalman filtering and ReID features"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from collections import deque
from enum import IntEnum

try:
    import scipy.linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some features may not work.")

try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. ReID transforms may not work.")


class TrackState(IntEnum):
    """Enumeration type for track state"""
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class KalmanFilter:
    """
    Kalman Filter for bounding box tracking
    State: [x, y, a, h, vx, vy, va, vh]
    where (x, y) is center, a is aspect ratio, h is height
    """
    def __init__(self):
        ndim, dt = 4, 1.
        
        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # Measurement matrix
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """Create track from unassociated measurement"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        if SCIPY_AVAILABLE:
            try:
                chol_factor, lower = scipy.linalg.cho_factor(
                    projected_cov, lower=True, check_finite=False)
                kalman_gain = scipy.linalg.cho_solve(
                    (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                    check_finite=False).T
            except Exception:
                # Fallback to numpy if scipy fails
                kalman_gain = np.linalg.solve(
                    projected_cov, np.dot(covariance, self._update_mat.T).T).T
        else:
            # Use numpy only
            kalman_gain = np.linalg.solve(
                projected_cov, np.dot(covariance, self._update_mat.T).T).T
        
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements):
        """Compute gating distance between state and measurements"""
        mean, covariance = self.project(mean, covariance)
        
        d = measurements - mean
        
        if SCIPY_AVAILABLE:
            try:
                cholesky_factor = np.linalg.cholesky(covariance)
                z = scipy.linalg.solve_triangular(
                    cholesky_factor, d.T, lower=True, check_finite=False,
                    overwrite_b=True)
            except Exception:
                # Fallback
                z = np.linalg.solve(cholesky_factor, d.T).T
        else:
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T).T
        
        squared_maha = np.sum(z * z, axis=0)
        
        return squared_maha


class Track:
    """
    A single track with Kalman filtering and feature history
    """
    def __init__(self, detection, track_id, n_init=3, max_age=30, feature=None):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = deque(maxlen=200)  # Feature gallery (increased for better matching)
        
        self.n_init = n_init  # Number of consecutive detections before confirmed
        self.max_age = max_age  # Max frames to keep alive without detection
        
        # Initialize Kalman filter
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection)
        
        # Add first feature
        if feature is not None:
            self.features.append(feature)
    
    def predict(self):
        """Propagate the state distribution to the current time step"""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection, feature=None):
        """Perform Kalman filter measurement update and feature update"""
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection)
        
        # Add feature to gallery
        if feature is not None:
            self.features.append(feature)
        
        self.hits += 1
        self.time_since_update = 0
        
        # Promote tentative tracks
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """Mark track as missed (no detection association)"""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted
    
    def is_confirmed(self):
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        return self.state == TrackState.Deleted
    
    def to_tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)"""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def to_tlbr(self):
        """Get current position in bounding box format (min x, min y, max x, max y)"""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret


class DeepSORT:
    """
    DeepSORT tracker with appearance features and Kalman filtering
    """
    def __init__(self, reid_model=None, max_dist=0.2, max_iou_distance=0.7, 
                 max_age=30, n_init=3, nn_budget=100):
        """
        Args:
            reid_model: ReID model for feature extraction (optional)
            max_dist: Maximum cosine distance for matching
            max_iou_distance: Maximum IOU distance for matching
            max_age: Maximum number of missed frames before track deletion
            n_init: Number of consecutive detections before track confirmation
            nn_budget: Maximum size of feature gallery
        """
        self.reid_model = reid_model
        if reid_model is not None:
            self.reid_model.eval()
            self.device = next(reid_model.parameters()).device
        else:
            self.device = torch.device('cpu')
        
        self.max_dist = max_dist
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        
        self.tracks = []
        self.next_id = 1
        
        # Transform for ReID input
        if TORCHVISION_AVAILABLE:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = None
    
    def update(self, detections, frame):
        """
        Update tracks with new detections
        
        Args:
            detections: List/array of bounding boxes [x1, y1, x2, y2, confidence]
            frame: Current frame (numpy array) for feature extraction
        
        Returns:
            List of active tracks with [x1, y1, x2, y2, track_id]
        """
        if len(detections) == 0:
            detections = np.empty((0, 5))
        else:
            detections = np.array(detections)
        
        # Filter by confidence - match detection threshold (0.15)
        if len(detections) > 0 and detections.shape[1] >= 5:
            min_conf = 0.15  # Match detection threshold to avoid discarding detections
            detections = detections[detections[:, 4] >= min_conf]
        
        # Extract appearance features
        features = self._extract_features(detections, frame)
        
        # Convert detections to [cx, cy, aspect_ratio, height]
        detection_measurements = self._detections_to_measurements(detections)
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks using cascade matching (Priority 4)
        matched, unmatched_tracks, unmatched_detections = self._cascade_match(
            detection_measurements, features, detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matched:
            feature = features[detection_idx] if len(features) > 0 else None
            self.tracks[track_idx].update(
                detection_measurements[detection_idx],
                feature
            )
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            feature = features[detection_idx] if len(features) > 0 else None
            self._initiate_track(
                detection_measurements[detection_idx],
                feature
            )
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return active tracks
        results = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlbr()
                results.append([*bbox, track.track_id])
        
        return results
    
    def _extract_features(self, detections, frame):
        """Extract ReID features from detections"""
        # If ReID model available, use it
        if self.reid_model is not None and self.transform is not None and len(detections) > 0:
            return self._extract_reid_features(detections, frame)
        
        # Otherwise, use simple color histogram as appearance feature
        if len(detections) > 0:
            return self._extract_color_features(detections, frame)
        
        return np.array([])
    
    def _extract_reid_features(self, detections, frame):
        """Extract features using ReID model"""
        
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                try:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_tensor = self.transform(crop)
                        crops.append(crop_tensor)
                    else:
                        crops.append(torch.zeros((3, 256, 128)))
                except Exception:
                    crops.append(torch.zeros((3, 256, 128)))
            else:
                # Invalid crop, use zero tensor
                crops.append(torch.zeros((3, 256, 128)))
        
        if not crops:
            return np.array([])
        
        # Batch process
        crops = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            features = self.reid_model(crops, return_features=True)
        
        return features.cpu().numpy()
    
    def _extract_color_features(self, detections, frame):
        """Extract enhanced color histogram features with spatial info"""
        import cv2
        
        features = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    h, w = crop.shape[:2]
                    
                    # Split into top/bottom for better discrimination (head vs body)
                    top_half = crop[:h//2, :]
                    bottom_half = crop[h//2:, :]
                    
                    # Extract features for each part
                    def extract_part_features(part):
                        if part.size == 0:
                            return np.zeros(80)  # 32+32+16
                        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
                        # More bins for better discrimination
                        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
                        
                        # Normalize
                        hist_h = hist_h / (hist_h.sum() + 1e-8)
                        hist_s = hist_s / (hist_s.sum() + 1e-8)
                        hist_v = hist_v / (hist_v.sum() + 1e-8)
                        
                        hist = np.concatenate((
                            hist_h.flatten(),
                            hist_s.flatten(),
                            hist_v.flatten()
                        ))
                        return hist
                    
                    # Combine top and bottom features
                    top_feat = extract_part_features(top_half)
                    bottom_feat = extract_part_features(bottom_half)
                    
                    # Weight top more (face/head more discriminative)
                    combined = np.concatenate([top_feat * 0.6, bottom_feat * 0.4])
                    # L2 normalize for cosine similarity
                    combined = combined / (np.linalg.norm(combined) + 1e-8)
                    features.append(combined)
                else:
                    features.append(np.zeros(160))  # 80*2 for top+bottom
            else:
                features.append(np.zeros(160))
        
        return np.array(features) if features else np.array([])
    
    def _detections_to_measurements(self, detections):
        """Convert [x1, y1, x2, y2, conf] to [cx, cy, aspect_ratio, height]"""
        if len(detections) == 0:
            return np.empty((0, 4))
        
        measurements = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = w / h if h > 0 else 1.0
            measurements.append([cx, cy, aspect_ratio, h])
        
        return np.array(measurements)
    
    def _match(self, detections, features):
        """
        Match detections to tracks using appearance and motion
        
        Returns:
            matched: List of (track_idx, detection_idx)
            unmatched_tracks: List of track indices
            unmatched_detections: List of detection indices
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(detections, features)
        
        # Apply Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out matches with high cost
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            cost = cost_matrix[track_idx, detection_idx]
            # Use adaptive threshold: stricter for confirmed tracks, looser for tentative
            track = self.tracks[track_idx]
            # For confirmed tracks, use strict threshold (0.6x) to prevent ID switches
            # But not too strict to avoid breaking valid matches
            # For tentative tracks, use looser threshold (1.3x) to allow initial matching
            threshold = self.max_dist * 0.6 if track.is_confirmed() else self.max_dist * 1.3
            
            if cost > threshold:
                unmatched_tracks.append(track_idx)
            else:
                matches.append((track_idx, detection_idx))
                if detection_idx in unmatched_detections:
                    unmatched_detections.remove(detection_idx)
        
        # Add tracks that weren't matched at all
        matched_track_indices = set(track_indices)
        for track_idx in range(len(self.tracks)):
            if track_idx not in matched_track_indices:
                unmatched_tracks.append(track_idx)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _compute_cost_matrix(self, detections, features):
        """Compute cost matrix using appearance and motion"""
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            # Gating: use Mahalanobis distance for motion
            gating_distance = track.kf.gating_distance(
                track.mean, track.covariance, detections)
            
            # Appearance distance
            if len(features) > 0 and len(track.features) > 0:
                # Compute minimum cosine distance to feature gallery
                track_features = np.array(list(track.features))
                appearance_distance = self._compute_appearance_distance(
                    track_features, features)
                
                # Combine distances - STRONGLY prioritize appearance for better ID preservation
                # Use 0.2 motion + 0.8 appearance to strongly prioritize appearance matching
                # This reduces ID switches by relying more on appearance than motion
                cost_matrix[i] = 0.2 * gating_distance + 0.8 * appearance_distance
            else:
                # If no appearance features, use only motion (less reliable)
                cost_matrix[i] = gating_distance
            
            # Apply gate - use stricter threshold for better matching
            cost_matrix[i][gating_distance > 9.4877] = 1e5  # Chi-square threshold
        
        return cost_matrix
    
    def _compute_appearance_distance(self, track_features, detection_features):
        """Compute minimum cosine distance between track gallery and detections"""
        if len(track_features) == 0 or len(detection_features) == 0:
            return np.ones(len(detection_features)) * 1e5
        
        distances = []
        for det_feat in detection_features:
            # Compute cosine similarity with all features in gallery
            similarities = np.dot(track_features, det_feat)
            
            # Use weighted average of top-k similarities for more robust matching
            # This considers multiple similar features, not just the best one
            top_k = min(3, len(similarities))  # Use top 3 most similar features
            top_similarities = np.partition(similarities, -top_k)[-top_k:]
            avg_similarity = np.mean(top_similarities)
            
            # Convert to distance
            min_distance = 1 - avg_similarity
            distances.append(min_distance)
        
        return np.array(distances)
    
    def _cascade_match(self, detections, features, detection_boxes):
        """
        Cascade matching: first appearance+motion, then IoU for unmatched
        """
        # First pass: appearance + motion matching
        matches, unmatched_tracks, unmatched_detections = self._match(detections, features)
        
        # Second pass: IoU matching for unmatched
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            unmatched_track_objects = [self.tracks[i] for i in unmatched_tracks]
            # Get detection boxes for unmatched detections
            unmatched_det_boxes = detection_boxes[unmatched_detections] if len(detection_boxes) > 0 else np.array([])
            
            if len(unmatched_det_boxes) > 0:
                iou_matches = self._iou_match(unmatched_track_objects, unmatched_det_boxes)
                
                # Update matches - iou_matches contains (local_track_idx, local_det_idx)
                for track_local_idx, det_local_idx in iou_matches:
                    if track_local_idx < len(unmatched_tracks) and det_local_idx < len(unmatched_detections):
                        track_idx = unmatched_tracks[track_local_idx]
                        det_idx = unmatched_detections[det_local_idx]
                        matches.append((track_idx, det_idx))
                        # Remove from unmatched lists (in reverse order to avoid index issues)
                        if det_idx in unmatched_detections:
                            unmatched_detections.remove(det_idx)
                        if track_idx in unmatched_tracks:
                            unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _iou_match(self, tracks, detections):
        """Match using IoU for heavily occluded cases"""
        if len(tracks) == 0 or len(detections) == 0:
            return []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            track_bbox = track.to_tlbr()
            for j, det in enumerate(detections):
                det_bbox = det[:4]  # [x1, y1, x2, y2]
                iou_matrix[i, j] = self._compute_iou(track_bbox, det_bbox)
        
        # Use Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        for track_idx, det_idx in zip(track_indices, det_indices):
            if iou_matrix[track_idx, det_idx] > 0.25:  # Looser threshold for crowded scenes
                matches.append((track_idx, det_idx))
        
        return matches
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _initiate_track(self, measurement, feature):
        """Create new track"""
        track = Track(measurement, self.next_id, self.n_init, 
                     self.max_age, feature)
        self.tracks.append(track)
        self.next_id += 1


