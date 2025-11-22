"""
Training script for Transformer-based tracker
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import math

# Optional dependency
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Training will continue without logging.")

from src.models.tracking.transformer_tracker import TransformerTracker


class TrackingDataset(Dataset):
    """
    Dataset for training transformer tracker
    Creates sequences of detections and tracks from MOT20
    """
    def __init__(self, data_path: str, split: str = 'train', sequence_length: int = 10):
        """
        Args:
            data_path: Path to MOT20 dataset
            split: 'train' or 'val'
            sequence_length: Length of sequences to extract
        """
        self.data_path = Path(data_path)
        self.split = split
        self.sequence_length = sequence_length
        
        # Load MOT sequences with ground truth
        self.sequences = self._load_sequences()
    
    def _load_sequences(self) -> List[Dict]:
        """Load MOT20 sequences with tracking annotations"""
        sequences = []
        
        split_path = self.data_path / self.split
        if not split_path.exists():
            print(f"Warning: Split path {split_path} does not exist")
            return sequences
        
        seq_dirs = list(split_path.glob('*'))
        if not seq_dirs:
            print(f"Warning: No sequences found in {split_path}")
            return sequences
        
        for seq_dir in seq_dirs:
            if not seq_dir.is_dir():
                continue
            
            gt_file = seq_dir / 'gt' / 'gt.txt'
            if not gt_file.exists():
                continue
            
            # Parse ground truth
            frame_data = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    
                    try:
                        frame = int(parts[0])
                        track_id = int(parts[1])
                        x, y, w, h = map(float, parts[2:6])
                        visibility = float(parts[8])
                        
                        if visibility > 0.3:
                            if frame not in frame_data:
                                frame_data[frame] = {'detections': []}
                            
                            frame_data[frame]['detections'].append({
                                'bbox': [x, y, x+w, y+h],
                                'track_id': track_id
                            })
                    except (ValueError, IndexError):
                        continue
            
            # Create sequences
            sorted_frames = sorted(frame_data.keys())
            if len(sorted_frames) < self.sequence_length:
                continue
            
            for i in range(0, len(sorted_frames) - self.sequence_length, 5):
                seq_frames = sorted_frames[i:i+self.sequence_length]
                sequences.append({
                    'sequence': seq_dir.name,
                    'frames': seq_frames,
                    'data': frame_data
                })
        
        print(f"Loaded {len(sequences)} sequences from {self.split}")
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        
        # Collect detections and ground truth associations
        detections_list = []
        tracks_list = []
        associations = []
        
        for frame_idx, frame in enumerate(seq['frames']):
            if frame not in seq['data']:
                continue
            
            frame_data = seq['data'][frame]
            
            # Current detections
            dets = [d['bbox'] for d in frame_data['detections']]
            det_ids = [d['track_id'] for d in frame_data['detections']]
            
            # Previous tracks (from last frame)
            if frame_idx > 0:
                prev_frame = seq['frames'][frame_idx - 1]
                if prev_frame in seq['data']:
                    tracks = [d['bbox'] for d in seq['data'][prev_frame]['detections']]
                    track_ids = [d['track_id'] for d in seq['data'][prev_frame]['detections']]
                    
                    # Create association matrix
                    assoc = np.zeros((len(tracks), len(dets)), dtype=np.float32)
                    for i, tid in enumerate(track_ids):
                        for j, did in enumerate(det_ids):
                            if tid == did:
                                assoc[i, j] = 1.0
                    
                    detections_list.append(np.array(dets, dtype=np.float32) if dets else np.empty((0, 4), dtype=np.float32))
                    tracks_list.append(np.array(tracks, dtype=np.float32) if tracks else np.empty((0, 4), dtype=np.float32))
                    associations.append(assoc)
        
        return {
            'detections': detections_list,
            'tracks': tracks_list,
            'associations': associations
        }


class TransformerTrackerLoss(nn.Module):
    """Combined loss for transformer tracker training"""
    def __init__(self, lambda_match: float = 2.0, lambda_bbox: float = 1.0, lambda_state: float = 0.5):
        """
        Args:
            lambda_match: Weight for match loss
            lambda_bbox: Weight for bbox loss
            lambda_state: Weight for state loss
        """
        super().__init__()
        self.lambda_match = lambda_match
        self.lambda_bbox = lambda_bbox
        self.lambda_state = lambda_state
    
    def forward(self, pred_matches: torch.Tensor, pred_bbox: torch.Tensor, pred_states: torch.Tensor,
                gt_associations: torch.Tensor, gt_bbox: torch.Tensor, gt_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_matches: (B, N_track, N_det) predicted match scores
            pred_bbox: (B, N_track, 4) predicted refined boxes
            pred_states: (B, N_track, 3) predicted track states
            gt_associations: (B, N_track, N_det) ground truth associations
            gt_bbox: (B, N_track, 4) ground truth boxes
            gt_states: (B, N_track) ground truth states (0=inactive, 1=active, 2=new)
        
        Returns:
            total_loss, match_loss, bbox_loss, state_loss
        """
        # Match loss (binary cross entropy)
        match_loss = F.binary_cross_entropy_with_logits(
            pred_matches, gt_associations, reduction='mean'
        )
        
        # Bounding box regression loss (smooth L1)
        # Only compute for active tracks
        active_mask = (gt_states == 1).unsqueeze(-1).expand_as(pred_bbox)
        if active_mask.sum() > 0:
            bbox_loss = F.smooth_l1_loss(
                pred_bbox[active_mask].view(-1, 4),
                gt_bbox[active_mask].view(-1, 4),
                reduction='mean'
            )
        else:
            bbox_loss = torch.tensor(0.0, device=pred_bbox.device)
        
        # State classification loss (cross entropy)
        state_loss = F.cross_entropy(
            pred_states.view(-1, 3),
            gt_states.view(-1).long(),
            reduction='mean'
        )
        
        # Combined loss
        total_loss = (self.lambda_match * match_loss +
                     self.lambda_bbox * bbox_loss +
                     self.lambda_state * state_loss)
        
        return total_loss, match_loss, bbox_loss, state_loss


class TransformerTrackerTrainer:
    """Trainer for transformer-based tracker"""
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize W&B
        if WANDB_AVAILABLE:
            wandb.init(
                project="people-tracking-transformer",
                config=config,
                name="transformer-tracker-mot20"
            )
        
        # Create model
        self.model = TransformerTracker(
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ).to(self.device)
        
        # Loss function
        self.criterion = TransformerTrackerLoss(
            lambda_match=config['training']['lambda_match'],
            lambda_bbox=config['training']['lambda_bbox'],
            lambda_state=config['training']['lambda_state']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['min_lr']
        )
        
        # Dataset
        self.train_dataset = TrackingDataset(
            data_path=config['data']['path'],
            split='train',
            sequence_length=config['data']['sequence_length']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4),
            collate_fn=self._collate_fn
        )
        
        # Create checkpoints directory
        Path('checkpoints').mkdir(exist_ok=True)
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for variable-length sequences"""
        # Find max lengths
        max_det = max([len(seq['detections']) for seq in batch] + [1])
        max_track = max([len(seq['tracks']) for seq in batch] + [1])
        max_seq_len = max([len(seq['detections']) for seq in batch] + [1])
        
        batch_size = len(batch)
        
        # Initialize padded tensors
        detections_bbox = torch.zeros(batch_size, max_seq_len, max_det, 4)
        detections_feat = torch.zeros(batch_size, max_seq_len, max_det, 2048)  # Placeholder features
        tracks_bbox = torch.zeros(batch_size, max_seq_len, max_track, 4)
        tracks_feat = torch.zeros(batch_size, max_seq_len, max_track, 2048)  # Placeholder features
        associations = torch.zeros(batch_size, max_seq_len, max_track, max_det)
        gt_states = torch.zeros(batch_size, max_seq_len, max_track, dtype=torch.long)
        
        # Fill tensors
        for b, seq in enumerate(batch):
            seq_len = len(seq['detections'])
            for t in range(seq_len):
                if t < len(seq['detections']):
                    dets = seq['detections'][t]
                    if len(dets) > 0:
                        detections_bbox[b, t, :len(dets)] = torch.from_numpy(dets)
                
                if t < len(seq['tracks']):
                    tracks = seq['tracks'][t]
                    if len(tracks) > 0:
                        tracks_bbox[b, t, :len(tracks)] = torch.from_numpy(tracks)
                        # Set active state for existing tracks
                        gt_states[b, t, :len(tracks)] = 1
                
                if t < len(seq['associations']):
                    assoc = seq['associations'][t]
                    if assoc.size > 0:
                        h, w = assoc.shape
                        associations[b, t, :h, :w] = torch.from_numpy(assoc)
        
        return {
            'detections_bbox': detections_bbox,
            'detections_feat': detections_feat,
            'tracks_bbox': tracks_bbox,
            'tracks_feat': tracks_feat,
            'associations': associations,
            'gt_states': gt_states
        }
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_match_loss = 0
        total_bbox_loss = 0
        total_state_loss = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            detections_bbox = batch['detections_bbox'].to(self.device)
            detections_feat = batch['detections_feat'].to(self.device)
            tracks_bbox = batch['tracks_bbox'].to(self.device)
            tracks_feat = batch['tracks_feat'].to(self.device)
            gt_associations = batch['associations'].to(self.device)
            gt_states = batch['gt_states'].to(self.device)
            
            # Process sequence (for now, use first frame)
            # In practice, you'd process the full sequence
            seq_idx = 0
            det_bbox = detections_bbox[:, seq_idx]
            det_feat = detections_feat[:, seq_idx]
            track_bbox = tracks_bbox[:, seq_idx]
            track_feat = tracks_feat[:, seq_idx]
            gt_assoc = gt_associations[:, seq_idx]
            gt_state = gt_states[:, seq_idx]
            
            # Forward pass
            match_scores, refined_bbox, track_states = self.model(
                det_bbox, det_feat, track_bbox, track_feat
            )
            
            # Calculate loss
            loss, match_loss, bbox_loss, state_loss = self.criterion(
                match_scores, refined_bbox, track_states,
                gt_assoc, track_bbox, gt_state
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_match_loss += match_loss.item()
            total_bbox_loss += bbox_loss.item()
            total_state_loss += state_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'match': match_loss.item(),
                'bbox': bbox_loss.item(),
                'state': state_loss.item()
            })
            
            # Log to W&B
            if WANDB_AVAILABLE and batch_idx % 10 == 0:
                wandb.log({
                    'train/total_loss': loss.item(),
                    'train/match_loss': match_loss.item(),
                    'train/bbox_loss': bbox_loss.item(),
                    'train/state_loss': state_loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_match = total_match_loss / len(self.train_loader)
        avg_bbox = total_bbox_loss / len(self.train_loader)
        avg_state = total_state_loss / len(self.train_loader)
        
        return avg_loss, avg_match, avg_bbox, avg_state
    
    def train(self):
        """Full training loop"""
        best_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            avg_loss, avg_match, avg_bbox, avg_state = self.train_epoch(epoch)
            
            print(f'Epoch {epoch}: Loss={avg_loss:.4f} (match={avg_match:.4f}, '
                  f'bbox={avg_bbox:.4f}, state={avg_state:.4f})')
            
            # Save checkpoint
            if epoch % 10 == 0:
                checkpoint_path = f'checkpoints/transformer_tracker_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': self.config
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = 'checkpoints/transformer_tracker_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': self.config
                }, best_path)
                print(f"Best model saved to {best_path}")
            
            self.scheduler.step()


# Default configuration
default_config = {
    'data': {
        'path': 'data/MOT20',
        'sequence_length': 10
    },
    'model': {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1
    },
    'training': {
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'min_lr': 1e-6,
        'lambda_match': 2.0,
        'lambda_bbox': 1.0,
        'lambda_state': 0.5,
        'num_workers': 4
    }
}


if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Train Transformer Tracker')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = default_config
        print("Using default configuration")
    
    trainer = TransformerTrackerTrainer(config)
    trainer.train()


