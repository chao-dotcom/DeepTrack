"""Utilities for working with MOT Challenge dataset"""
import os
import cv2
from pathlib import Path
from typing import Optional, Tuple


def get_mot_sequences(mot_dir: str) -> list:
    """Get list of available MOT sequences"""
    mot_path = Path(mot_dir)
    sequences = []
    
    # Check train and test directories
    for split in ['train', 'test']:
        split_dir = mot_path / split
        if split_dir.exists():
            for seq_dir in split_dir.iterdir():
                if seq_dir.is_dir() and (seq_dir / 'img1').exists():
                    sequences.append({
                        'name': seq_dir.name,
                        'path': str(seq_dir),
                        'split': split,
                        'img_dir': str(seq_dir / 'img1')
                    })
    
    return sequences


def mot_sequence_to_video(seq_path: str, output_path: str, fps: Optional[float] = None) -> bool:
    """Convert MOT image sequence to video file"""
    seq_dir = Path(seq_path)
    img_dir = seq_dir / 'img1'
    seqinfo_file = seq_dir / 'seqinfo.ini'
    
    if not img_dir.exists():
        print(f"Error: img1 directory not found in {seq_path}")
        return False
    
    # Read sequence info if available
    if seqinfo_file.exists():
        seq_info = {}
        with open(seqinfo_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    seq_info[key.strip()] = value.strip()
        
        if fps is None:
            fps = float(seq_info.get('frameRate', 30))
        width = int(seq_info.get('imWidth', 1920))
        height = int(seq_info.get('imHeight', 1080))
    else:
        # Default values if seqinfo.ini not found
        if fps is None:
            fps = 30.0
        # Get dimensions from first image
        img_files = sorted(img_dir.glob('*.jpg'))
        if not img_files:
            print(f"Error: No images found in {img_dir}")
            return False
        first_img = cv2.imread(str(img_files[0]))
        height, width = first_img.shape[:2]
    
    # Get all image files
    img_files = sorted(img_dir.glob('*.jpg'))
    if not img_files:
        print(f"Error: No images found in {img_dir}")
        return False
    
    print(f"Converting {len(img_files)} images to video...")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for i, img_file in enumerate(img_files):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
        
        # Resize if needed
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(img_files)} frames...")
    
    out.release()
    print(f"Video saved to: {output_path}")
    return True


def list_mot_sequences(mot_dir: str = "data/raw/MOT20/MOT20"):
    """List all available MOT sequences"""
    sequences = get_mot_sequences(mot_dir)
    
    if not sequences:
        print(f"No MOT sequences found in {mot_dir}")
        return
    
    print(f"\nFound {len(sequences)} MOT sequences:\n")
    print(f"{'Name':<15} {'Split':<10} {'Path':<50}")
    print("-" * 75)
    
    for seq in sequences:
        print(f"{seq['name']:<15} {seq['split']:<10} {seq['path']:<50}")
    
    return sequences

