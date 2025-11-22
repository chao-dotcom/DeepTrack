"""MOT20 dataset for YOLO detection training"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Augmentation will be limited.")


class MOT20Dataset(Dataset):
    """
    Custom MOT20 dataset for YOLO training
    Handles dense crowds and occlusions
    """
    def __init__(self, data_path: str, split: str = 'train', augment: bool = True):
        self.data_path = Path(data_path)
        self.split = split
        self.augment = augment
        
        # Advanced augmentation for crowded scenes
        if ALBUMENTATIONS_AVAILABLE and augment:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = None
        
        self.images, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[List[str], List[List[Dict]]]:
        """Load MOT20 format data"""
        images = []
        labels = []
        
        split_path = self.data_path / self.split
        if not split_path.exists():
            print(f"Warning: Split path {split_path} does not exist")
            return images, labels
        
        sequences = list(split_path.glob('*'))
        if not sequences:
            print(f"Warning: No sequences found in {split_path}")
            return images, labels
        
        for seq in sequences:
            if not seq.is_dir():
                continue
                
            img_dir = seq / 'img1'
            gt_file = seq / 'gt' / 'gt.txt'
            
            if not img_dir.exists() or not gt_file.exists():
                continue
            
            # Parse ground truth
            frame_data = self._parse_mot_gt(gt_file, img_dir)
            
            for frame_num, bboxes in frame_data.items():
                img_path = img_dir / f'{frame_num:06d}.jpg'
                if img_path.exists() and len(bboxes) > 0:
                    images.append(str(img_path))
                    labels.append(bboxes)
        
        print(f"Loaded {len(images)} images from {len(sequences)} sequences")
        return images, labels
    
    def _parse_mot_gt(self, gt_file: Path, img_dir: Path) -> Dict[int, List[Dict]]:
        """Parse MOT20 ground truth format"""
        frame_data = {}
        
        if not gt_file.exists():
            return frame_data
        
        # Get image dimensions from first frame
        first_img = img_dir / '000001.jpg'
        if first_img.exists():
            img = cv2.imread(str(first_img))
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = 1920, 1080  # Default MOT20 resolution
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                
                try:
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])
                    conf = float(parts[6])
                    visibility = float(parts[8])
                    
                    # Filter out low visibility and low confidence
                    if conf > 0 and visibility > 0.3 and w > 0 and h > 0:
                        # Convert to YOLO format (normalized center_x, center_y, w, h)
                        center_x = (x + w / 2) / img_w
                        center_y = (y + h / 2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        
                        if frame not in frame_data:
                            frame_data[frame] = []
                        
                        frame_data[frame].append({
                            'bbox': [center_x, center_y, norm_w, norm_h],
                            'class': 0,  # Person class
                            'visibility': visibility
                        })
                except (ValueError, IndexError) as e:
                    continue
        
        return frame_data
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        bboxes = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return dummy data if image can't be loaded
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if enabled
        if self.transform is not None and self.augment:
            try:
                bbox_list = [b['bbox'] for b in bboxes]
                class_labels = [b['class'] for b in bboxes]
                
                transformed = self.transform(
                    image=image,
                    bboxes=bbox_list,
                    class_labels=class_labels
                )
                image = transformed['image']
                bboxes = [
                    {'bbox': bbox, 'class': cls}
                    for bbox, cls in zip(transformed['bboxes'], transformed['class_labels'])
                ]
            except Exception as e:
                # If augmentation fails, use original
                pass
        
        return image, bboxes


