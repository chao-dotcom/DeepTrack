"""ReID model training script"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from pathlib import Path
import numpy as np
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Training will proceed without logging.")

from src.data.reid_dataset import MOT20ReIDDataset
from src.models.reid.reid_model import ReIDModel, CombinedLoss


class ReIDTrainer:
    """Trainer for ReID model"""
    def __init__(self, config_path: str = None, config: dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # Default config
            self.config = {
                'data': {
                    'path': 'data/raw/MOT20/MOT20'
                },
                'model': {
                    'feature_dim': 2048,
                    'dropout': 0.5
                },
                'training': {
                    'epochs': 120,
                    'batch_size': 32,
                    'learning_rate': 0.00035,
                    'weight_decay': 0.0005,
                    'lr_step': 40,
                    'lr_gamma': 0.1,
                    'triplet_margin': 0.3,
                    'lambda_triplet': 1.0,
                    'lambda_cls': 1.0
                }
            }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize W&B
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="people-tracking-reid",
                    config=self.config,
                    name="resnet50-triplet-mot20"
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
        
        # Create datasets
        data_path = self.config['data']['path']
        print(f"Loading datasets from: {data_path}")
        
        try:
            self.train_dataset = MOT20ReIDDataset(
                data_path=data_path,
                split='train',
                triplet=True
            )
            
            # For validation, use a subset of train or create separate val set
            self.val_dataset = MOT20ReIDDataset(
                data_path=data_path,
                split='train',  # Use train split for now
                triplet=False
            )
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
        
        if len(self.train_dataset.person_ids) == 0:
            raise ValueError("No person data found! Please check data path and MOT20 dataset.")
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Create model
        num_classes = len(self.train_dataset.person_ids)
        print(f"Number of person classes: {num_classes}")
        
        self.model = ReIDModel(
            num_classes=num_classes,
            feature_dim=self.config['model']['feature_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Loss function
        self.criterion = CombinedLoss(
            num_classes=num_classes,
            margin=self.config['training']['triplet_margin'],
            lambda_triplet=self.config['training']['lambda_triplet'],
            lambda_cls=self.config['training']['lambda_cls']
        )
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config['training']['lr_step'],
            gamma=self.config['training']['lr_gamma']
        )
        
        # Create checkpoints directory
        Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_triplet = 0
        total_cls = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            anchor_logits, anchor_features = self.model(anchor)
            anchor_norm = self.model(anchor, return_features=True)
            positive_norm = self.model(positive, return_features=True)
            negative_norm = self.model(negative, return_features=True)
            
            # Calculate loss
            loss, triplet_loss, cls_loss = self.criterion(
                anchor_logits, anchor_features, labels,
                anchor_norm, positive_norm, negative_norm
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_triplet += triplet_loss.item()
            total_cls += cls_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'triplet': f'{triplet_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })
            
            # Log to W&B
            if WANDB_AVAILABLE and batch_idx % 10 == 0:
                try:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/triplet_loss': triplet_loss.item(),
                        'train/cls_loss': cls_loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr']
                    })
                except Exception:
                    pass
        
        avg_loss = total_loss / len(self.train_loader)
        avg_triplet = total_triplet / len(self.train_loader)
        avg_cls = total_cls / len(self.train_loader)
        
        return avg_loss, avg_triplet, avg_cls
    
    def validate(self):
        """Validate using rank-1 accuracy and mAP"""
        self.model.eval()
        
        # Extract all features
        gallery_features = []
        gallery_labels = []
        query_features = []
        query_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Extracting features'):
                images = images.to(self.device)
                features = self.model(images, return_features=True)
                
                # Split into gallery and query
                split = len(features) // 2
                gallery_features.append(features[:split].cpu())
                gallery_labels.extend(labels[:split].tolist())
                query_features.append(features[split:].cpu())
                query_labels.extend(labels[split:].tolist())
        
        if len(gallery_features) == 0:
            return 0.0, 0.0
        
        gallery_features = torch.cat(gallery_features, dim=0)
        query_features = torch.cat(query_features, dim=0)
        
        # Calculate distances
        dist_matrix = self.compute_distance_matrix(query_features, gallery_features)
        
        # Calculate metrics
        rank1, mAP = self.evaluate_rank(dist_matrix, query_labels, gallery_labels)
        
        if WANDB_AVAILABLE:
            try:
                wandb.log({
                    'val/rank1': rank1,
                    'val/mAP': mAP
                })
            except Exception:
                pass
        
        return rank1, mAP
    
    def compute_distance_matrix(self, query, gallery):
        """Compute cosine distance matrix"""
        # Cosine similarity
        similarity = torch.mm(query, gallery.t())
        # Convert to distance
        distance = 1 - similarity
        return distance.numpy()
    
    def evaluate_rank(self, dist_matrix, query_labels, gallery_labels):
        """Calculate rank-1 accuracy and mAP"""
        if dist_matrix.size == 0:
            return 0.0, 0.0
        
        num_query = dist_matrix.shape[0]
        
        # Sort distances
        indices = np.argsort(dist_matrix, axis=1)
        
        # Calculate rank-1
        rank1_correct = 0
        aps = []
        
        for i in range(num_query):
            query_label = query_labels[i]
            
            # Get sorted gallery labels
            sorted_gallery_labels = [gallery_labels[idx] for idx in indices[i]]
            
            # Rank-1: first match
            if sorted_gallery_labels[0] == query_label:
                rank1_correct += 1
            
            # mAP: average precision
            matches = [1 if label == query_label else 0 for label in sorted_gallery_labels]
            ap = self.calculate_ap(matches)
            aps.append(ap)
        
        rank1 = rank1_correct / num_query if num_query > 0 else 0.0
        mAP = np.mean(aps) if aps else 0.0
        
        return rank1, mAP
    
    def calculate_ap(self, matches):
        """Calculate average precision"""
        if sum(matches) == 0:
            return 0.0
        
        precisions = []
        num_correct = 0
        
        for i, match in enumerate(matches):
            if match:
                num_correct += 1
                precision = num_correct / (i + 1)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def train(self):
        """Full training loop"""
        best_rank1 = 0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            avg_loss, avg_triplet, avg_cls = self.train_epoch(epoch)
            
            print(f'Epoch {epoch}: Loss={avg_loss:.4f}, '
                  f'Triplet={avg_triplet:.4f}, Cls={avg_cls:.4f}')
            
            # Validate every 5 epochs
            if epoch % 5 == 0:
                rank1, mAP = self.validate()
                print(f'Validation: Rank-1={rank1:.4f}, mAP={mAP:.4f}')
                
                # Save best model
                if rank1 > best_rank1:
                    best_rank1 = rank1
                    checkpoint_path = 'models/checkpoints/reid_best.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'rank1': rank1,
                        'mAP': mAP,
                        'num_classes': len(self.train_dataset.person_ids)
                    }, checkpoint_path)
                    print(f'Saved best model to {checkpoint_path} with Rank-1: {rank1:.4f}')
            
            # Step scheduler
            self.scheduler.step()
        
        print(f'Training complete! Best Rank-1: {best_rank1:.4f}')
        return best_rank1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ReID model')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data-path', type=str, help='Path to MOT20 dataset')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ReIDTrainer(config_path=args.config)
    
    # Override config if provided
    if args.data_path:
        trainer.config['data']['path'] = args.data_path
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    
    # Train
    best_rank1 = trainer.train()


