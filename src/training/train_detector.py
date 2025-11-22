"""Detection model training script"""
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Training will proceed without logging.")


class DetectionTrainer:
    """Trainer for YOLOv8 detection model on MOT20"""
    def __init__(self, config_path: str = None, config: dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # Default config
            self.config = {
                'model': {
                    'architecture': 'yolov8n',  # Use nano for faster training
                    'pretrained': True
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 16,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0005,
                    'warmup_epochs': 3,
                    'box_loss_weight': 7.5,
                    'cls_loss_weight': 0.5,
                    'dfl_loss_weight': 1.5
                },
                'augmentation': {
                    'mosaic': 0.5,
                    'mixup': 0.2,
                    'degrees': 10.0,
                    'translate': 0.2,
                    'scale': 0.5,
                    'fliplr': 0.5
                }
            }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize W&B if available
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="people-tracking-detection",
                    config=self.config,
                    name="yolov8-mot20-finetune"
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                WANDB_AVAILABLE = False
        
        # Load model
        model_name = self.config['model']['architecture']
        if self.config['model'].get('pretrained', True):
            model_path = f'{model_name}.pt'
        else:
            model_path = f'{model_name}.yaml'
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to default YOLOv8n")
            self.model = YOLO('yolov8n.pt')
    
    def train(self, data_yaml: str = None):
        """Train detection model with custom configuration"""
        # Create data YAML if not provided
        if data_yaml is None:
            data_yaml = self._create_data_yaml()
        
        print(f"Training with data config: {data_yaml}")
        
        # Prepare training arguments
        train_args = {
            'data': data_yaml,
            'epochs': self.config['training']['epochs'],
            'imgsz': 640,  # Can be increased to 1280 for better accuracy
            'batch': self.config['training']['batch_size'],
            'device': str(self.device),
            'lr0': self.config['training']['learning_rate'],
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'box': self.config['training']['box_loss_weight'],
            'cls': self.config['training']['cls_loss_weight'],
            'dfl': self.config['training']['dfl_loss_weight'],
            'patience': 20,
            'save_period': 10,
            'workers': 4,
        }
        
        # Add augmentation parameters
        train_args.update(self.config.get('augmentation', {}))
        
        # Train
        try:
            results = self.model.train(**train_args)
            print("Training completed successfully!")
            return results
        except Exception as e:
            print(f"Training error: {e}")
            raise
    
    def _create_data_yaml(self) -> str:
        """Create YOLO data YAML file"""
        data_yaml_path = Path('configs/mot20.yaml')
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default MOT20 paths
        data_config = {
            'path': str(Path('data/raw/MOT20/MOT20').absolute()),
            'train': 'train',
            'val': 'train',  # Use train for validation if no separate val set
            'nc': 1,
            'names': ['person']
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        return str(data_yaml_path)
    
    def evaluate(self, data_yaml: str = None):
        """Evaluate model on validation set"""
        if data_yaml is None:
            data_yaml = self._create_data_yaml()
        
        results = self.model.val(data=data_yaml, device=str(self.device))
        
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        
        return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 detection model')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data', type=str, help='Path to data YAML file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DetectionTrainer(config_path=args.config)
    
    # Override epochs if provided
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    
    # Train
    results = trainer.train(data_yaml=args.data)
    
    # Evaluate
    print("\nEvaluating model...")
    trainer.evaluate(data_yaml=args.data)
    
    print(f"\nTraining complete! Model saved to: {trainer.model.trainer.save_dir}")


