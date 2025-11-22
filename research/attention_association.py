"""
Cross-attention mechanism for track-detection association
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossAttentionAssociation(nn.Module):
    """
    Use cross-attention to compute detection-track associations
    More flexible than cosine similarity
    """
    def __init__(self, d_model: int = 512, nhead: int = 8):
        """
        Args:
            d_model: Feature dimension
            nhead: Number of attention heads
        """
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Association scorer
        self.association_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, detection_features: torch.Tensor, track_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            detection_features: (B, N_det, d_model)
            track_features: (B, N_track, d_model)
        
        Returns:
            association_matrix: (B, N_track, N_det)
        """
        # Cross-attention: tracks attend to detections
        attended, attention_weights = self.cross_attention(
            query=track_features,
            key=detection_features,
            value=detection_features
        )
        
        # Residual connection
        track_features = self.norm1(track_features + attended)
        
        # Feed-forward
        ff_out = self.feed_forward(track_features)
        track_features = self.norm2(track_features + ff_out)
        
        # Compute association scores
        # Expand for pairwise comparison
        B, N_track, d = track_features.shape
        _, N_det, _ = detection_features.shape
        
        track_expanded = track_features.unsqueeze(2).expand(-1, -1, N_det, -1)
        det_expanded = detection_features.unsqueeze(1).expand(-1, N_track, -1, -1)
        
        # Concatenate and score
        pairs = torch.cat([track_expanded, det_expanded], dim=-1)
        association_scores = self.association_head(pairs).squeeze(-1)
        
        return association_scores


