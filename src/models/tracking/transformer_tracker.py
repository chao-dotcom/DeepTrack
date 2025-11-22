"""
Transformer-based Multi-Object Tracker
Uses self-attention to model track-to-track relationships
and cross-attention for detection-to-track association
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        
        Returns:
            Positionally encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTracker(nn.Module):
    """
    Transformer-based Multi-Object Tracker
    Uses self-attention to model track-to-track relationships
    and cross-attention for detection-to-track association
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input embeddings
        self.bbox_embedding = nn.Linear(4, d_model // 2)
        self.feature_embedding = nn.Linear(2048, d_model // 2)  # ReID features
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder (for detections)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder (for tracks)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output heads
        self.match_head = nn.Linear(d_model * 2, 1)  # Match score (concatenated features)
        self.bbox_head = nn.Linear(d_model, 4)   # Bbox refinement
        self.state_head = nn.Linear(d_model, 3)  # Track state (active/inactive/new)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, detections_bbox: torch.Tensor, detections_feat: torch.Tensor,
                tracks_bbox: torch.Tensor, tracks_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            detections_bbox: (B, N_det, 4) detection bounding boxes
            detections_feat: (B, N_det, 2048) detection ReID features
            tracks_bbox: (B, N_track, 4) track bounding boxes
            tracks_feat: (B, N_track, 2048) track ReID features
        
        Returns:
            match_matrix: (B, N_track, N_det) matching scores
            refined_bbox: (B, N_track, 4) refined track boxes
            track_states: (B, N_track, 3) track state probabilities
        """
        # Embed detections
        det_bbox_emb = self.bbox_embedding(detections_bbox)
        det_feat_emb = self.feature_embedding(detections_feat)
        det_emb = torch.cat([det_bbox_emb, det_feat_emb], dim=-1)
        det_emb = self.pos_encoding(det_emb)
        
        # Embed tracks
        track_bbox_emb = self.bbox_embedding(tracks_bbox)
        track_feat_emb = self.feature_embedding(tracks_feat)
        track_emb = torch.cat([track_bbox_emb, track_feat_emb], dim=-1)
        track_emb = self.pos_encoding(track_emb)
        
        # Encode detections
        det_encoded = self.encoder(det_emb)
        
        # Decode tracks with detection context
        track_decoded = self.decoder(track_emb, det_encoded)
        
        # Compute match scores
        # Expand for pairwise comparison
        N_track = track_decoded.size(1)
        N_det = det_encoded.size(1)
        
        track_expanded = track_decoded.unsqueeze(2).expand(-1, -1, N_det, -1)
        det_expanded = det_encoded.unsqueeze(1).expand(-1, N_track, -1, -1)
        
        # Concatenate and compute match score
        pairs = torch.cat([track_expanded, det_expanded], dim=-1)
        match_scores = self.match_head(pairs).squeeze(-1)  # (B, N_track, N_det)
        
        # Refine bounding boxes
        refined_bbox = self.bbox_head(track_decoded)  # (B, N_track, 4)
        refined_bbox = tracks_bbox + refined_bbox  # Residual connection
        
        # Predict track states
        track_states = self.state_head(track_decoded)  # (B, N_track, 3)
        track_states = F.softmax(track_states, dim=-1)
        
        return match_scores, refined_bbox, track_states


