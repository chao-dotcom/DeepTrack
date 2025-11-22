"""Person Re-Identification dataset from MOT20"""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np


class MOT20ReIDDataset(Dataset):
    """
    Person Re-Identification dataset from MOT20
    Creates triplets: anchor, positive, negative
    """
    def __init__(self, data_path: str, split: str = 'train', triplet: bool = True):
        self.data_path = Path(data_path)
        self.split = split
        self.triplet = triplet
        
        # ReID-specific transforms
        self.transform = T.Compose([
            T.Resize((256, 128)),  # Standard ReID size
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.person_images = self._extract_person_crops()
        self.person_ids = list(self.person_images.keys())
        
        print(f"ReID Dataset: {len(self.person_ids)} unique persons, "
              f"{sum(len(imgs) for imgs in self.person_images.values())} total images")
    
    def _extract_person_crops(self) -> dict:
        """Extract and save person crops from MOT20 sequences"""
        person_dict = defaultdict(list)
        
        split_path = self.data_path / self.split
        if not split_path.exists():
            print(f"Warning: Split path {split_path} does not exist")
            return person_dict
        
        sequences = list(split_path.glob('*'))
        for seq in sequences:
            if not seq.is_dir():
                continue
                
            img_dir = seq / 'img1'
            gt_file = seq / 'gt' / 'gt.txt'
            
            if not img_dir.exists() or not gt_file.exists():
                continue
            
            crop_dir = seq / 'crops'
            crop_dir.mkdir(exist_ok=True)
            
            # Parse ground truth and extract crops
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    
                    try:
                        frame = int(parts[0])
                        person_id = int(parts[1])
                        x, y, w, h = map(int, map(float, parts[2:6]))
                        visibility = float(parts[8])
                        
                        # Only use highly visible persons with reasonable size
                        if visibility > 0.5 and w > 30 and h > 60:
                            img_path = img_dir / f'{frame:06d}.jpg'
                            if img_path.exists():
                                crop_path = crop_dir / f'{frame:06d}_{person_id}.jpg'
                                
                                if not crop_path.exists():
                                    try:
                                        img = Image.open(img_path)
                                        # Ensure coordinates are within image bounds
                                        img_w, img_h = img.size
                                        x = max(0, min(x, img_w))
                                        y = max(0, min(y, img_h))
                                        w = min(w, img_w - x)
                                        h = min(h, img_h - y)
                                        
                                        if w > 0 and h > 0:
                                            crop = img.crop((x, y, x+w, y+h))
                                            crop.save(crop_path)
                                    except Exception as e:
                                        continue
                                
                                if crop_path.exists():
                                    person_dict[person_id].append(str(crop_path))
                    except (ValueError, IndexError) as e:
                        continue
        
        # Filter persons with at least 10 images
        person_dict = {k: v for k, v in person_dict.items() if len(v) >= 10}
        
        return person_dict
    
    def __len__(self):
        if self.triplet:
            return len(self.person_ids) * 100  # Epoch size
        else:
            return sum(len(imgs) for imgs in self.person_images.values())
    
    def __getitem__(self, idx):
        """Return triplet: anchor, positive, negative"""
        if self.triplet:
            if len(self.person_ids) == 0:
                # Return dummy data if no persons
                dummy_img = torch.zeros((3, 256, 128))
                return dummy_img, dummy_img, dummy_img, 0
            
            # Select anchor person
            anchor_id = random.choice(self.person_ids)
            anchor_img_path = random.choice(self.person_images[anchor_id])
            
            # Select positive (same person, different image)
            positive_img_path = random.choice(self.person_images[anchor_id])
            while positive_img_path == anchor_img_path and len(self.person_images[anchor_id]) > 1:
                positive_img_path = random.choice(self.person_images[anchor_id])
            
            # Select negative (different person)
            negative_id = random.choice(self.person_ids)
            while negative_id == anchor_id and len(self.person_ids) > 1:
                negative_id = random.choice(self.person_ids)
            negative_img_path = random.choice(self.person_images[negative_id])
            
            # Load and transform
            try:
                anchor = self.transform(Image.open(anchor_img_path).convert('RGB'))
                positive = self.transform(Image.open(positive_img_path).convert('RGB'))
                negative = self.transform(Image.open(negative_img_path).convert('RGB'))
            except Exception as e:
                # Return dummy if loading fails
                dummy_img = torch.zeros((3, 256, 128))
                return dummy_img, dummy_img, dummy_img, anchor_id
            
            return anchor, positive, negative, anchor_id
        else:
            # Single image mode for inference
            if len(self.person_ids) == 0:
                dummy_img = torch.zeros((3, 256, 128))
                return dummy_img, 0
            
            person_id = self.person_ids[idx % len(self.person_ids)]
            img_path = random.choice(self.person_images[person_id])
            
            try:
                img = self.transform(Image.open(img_path).convert('RGB'))
            except Exception as e:
                img = torch.zeros((3, 256, 128))
            
            return img, person_id


