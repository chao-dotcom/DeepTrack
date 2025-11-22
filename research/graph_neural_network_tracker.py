"""
Graph Neural Network-based Tracker
Represents tracking as a graph matching problem
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

# Optional dependency
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: torch_geometric not available. GNN tracker will not work.")
    # Create dummy classes
    class GCNConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    class Data:
        pass


class GNNTracker(nn.Module):
    """
    Graph Neural Network for multi-object tracking
    Nodes: Detections and tracks
    Edges: Potential associations
    """
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        """
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for GNN layers
        """
        super().__init__()
        
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN tracker. Install with: pip install torch-geometric")
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolutional layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Edge classifier (association score)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, feature_dim) node features
            edge_index: (2, E) edge connectivity
        
        Returns:
            edge_scores: (E,) association scores for each edge
        """
        # Encode nodes
        x = self.node_encoder(node_features)
        
        # Graph convolution
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Compute edge features
        edge_features = self._compute_edge_features(x, edge_index)
        
        # Classify edges
        edge_scores = self.edge_classifier(edge_features).squeeze()
        
        return edge_scores
    
    def _compute_edge_features(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute features for each edge"""
        # Concatenate features of connected nodes
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]
        
        edge_features = torch.cat([src_features, dst_features], dim=1)
        
        return edge_features


class GNNTrackingSystem:
    """Complete GNN-based tracking system"""
    def __init__(self, gnn_model: GNNTracker, reid_model=None):
        """
        Args:
            gnn_model: Trained GNN tracker model
            reid_model: ReID model for feature extraction
        """
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN tracking system.")
        
        self.gnn_model = gnn_model
        self.reid_model = reid_model
        
        if reid_model is not None:
            self.device = next(reid_model.parameters()).device
            self.reid_model.eval()
        else:
            self.device = torch.device('cpu')
        
        self.gnn_model.eval()
        self.gnn_model.to(self.device)
        
        self.tracks = []
        self.next_id = 1
        self.track_features = []  # Store features for each track
    
    def _extract_features(self, detections: List, frame: np.ndarray) -> torch.Tensor:
        """Extract features from detections"""
        if self.reid_model is None:
            # Return random features if no ReID model
            return torch.randn(len(detections), 256).to(self.device)
        
        features = []
        
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                features.append(torch.zeros(256))
                continue
            
            # Transform for ReID
            from torchvision import transforms
            from PIL import Image
            import cv2
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Convert BGR to RGB
            if len(crop.shape) == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = crop
            
            crop_tensor = transform(crop_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.reid_model(crop_tensor, return_features=True)
            
            features.append(feature.cpu()[0])
        
        return torch.stack(features).to(self.device)
    
    def _get_track_features(self) -> torch.Tensor:
        """Get features for existing tracks"""
        if len(self.track_features) == 0:
            return torch.empty((0, 256)).to(self.device)
        
        # Average features for each track
        track_feats = []
        for track_feat_list in self.track_features:
            if len(track_feat_list) > 0:
                avg_feat = torch.stack(track_feat_list).mean(dim=0)
                track_feats.append(avg_feat)
            else:
                track_feats.append(torch.zeros(256))
        
        return torch.stack(track_feats).to(self.device) if track_feats else torch.empty((0, 256)).to(self.device)
    
    def _build_graph(self, detection_features: torch.Tensor, track_features: torch.Tensor) -> Data:
        """Build bipartite graph for matching"""
        # Nodes: [detections, tracks]
        node_features = torch.cat([detection_features, track_features], dim=0)
        
        # Edges: All possible detection-track pairs
        num_det = len(detection_features)
        num_track = len(track_features)
        
        edge_list = []
        for i in range(num_det):
            for j in range(num_track):
                # Edge from detection i to track j+num_det
                edge_list.append([i, j + num_det])
                edge_list.append([j + num_det, i])  # Bidirectional
        
        if len(edge_list) == 0:
            # No edges, create empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Create PyG data object
        data = Data(
            x=node_features,
            edge_index=edge_index
        )
        
        return data
    
    def _solve_assignment(self, edge_scores: torch.Tensor, graph_data: Data) -> List[Tuple[int, int]]:
        """Solve assignment problem using Hungarian algorithm"""
        if len(edge_scores) == 0:
            return []
        
        from scipy.optimize import linear_sum_assignment
        
        num_det = len(graph_data.x) - len(self.track_features)
        num_track = len(self.track_features)
        
        if num_track == 0:
            return []
        
        # Build cost matrix
        cost_matrix = np.ones((num_track, num_det)) * 1e5  # High cost for no match
        
        edge_idx = 0
        for i in range(num_det):
            for j in range(num_track):
                # Get score for this edge
                if edge_idx < len(edge_scores):
                    score = edge_scores[edge_idx].item()
                    cost = 1 - score  # Convert score to cost
                    cost_matrix[j, i] = cost
                edge_idx += 2  # Skip bidirectional edge
        
        # Solve assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by threshold
        matches = []
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < 0.5:  # Threshold
                matches.append((t_idx, d_idx))
        
        return matches
    
    def update(self, detections: List, frame: np.ndarray) -> List:
        """Update tracks using GNN"""
        # Extract features
        detection_features = self._extract_features(detections, frame)
        track_features = self._get_track_features()
        
        # Build graph
        graph_data = self._build_graph(detection_features, track_features)
        
        if graph_data.edge_index.shape[1] == 0:
            # No edges, create new tracks for all detections
            matches = []
        else:
            # Predict associations
            with torch.no_grad():
                edge_scores = self.gnn_model(
                    graph_data.x, 
                    graph_data.edge_index
                )
            
            # Solve assignment problem
            matches = self._solve_assignment(edge_scores, graph_data)
        
        # Update tracks
        self._update_tracks(matches, detections, detection_features)
        
        return self.get_active_tracks()
    
    def _update_tracks(self, matches: List[Tuple[int, int]], detections: List, detection_features: torch.Tensor):
        """Update tracks based on matches"""
        matched_detections = set()
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            if track_idx < len(self.tracks) and det_idx < len(detections):
                # Update track
                det = detections[det_idx]
                self.tracks[track_idx] = [*det[:4], self.tracks[track_idx][4]]  # Keep track ID
                
                # Update features
                if track_idx < len(self.track_features):
                    self.track_features[track_idx].append(detection_features[det_idx].cpu())
                    if len(self.track_features[track_idx]) > 10:
                        self.track_features[track_idx].pop(0)
                
                matched_detections.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_id = self.next_id
                self.next_id += 1
                
                self.tracks.append([*det[:4], new_id])
                self.track_features.append([detection_features[det_idx].cpu()])
    
    def get_active_tracks(self) -> List:
        """Get currently active tracks"""
        return self.tracks.copy()


