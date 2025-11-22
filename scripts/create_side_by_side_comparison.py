"""
Create side-by-side comparison videos
Implements Priority 4 from docs/5.md
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple


def create_comparison_video(
    original_video: str,
    tracked_video: str,
    output_path: str,
    labels: Tuple[str, str] = ("Original", "Tracked")
):
    """
    Create side-by-side comparison video
    
    Args:
        original_video: Path to original video
        tracked_video: Path to tracked video
        output_path: Output video path
        labels: Labels for left and right videos
    """
    cap1 = cv2.VideoCapture(original_video)
    cap2 = cv2.VideoCapture(tracked_video)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files")
        return False
    
    # Get video properties
    fps = int(cap1.get(cv2.CAP_PROP_FPS)) or 25
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use max dimensions
    width = max(width1, width2)
    height = max(height1, height2)
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Resize frames to same size
        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))
        
        # Add labels
        cv2.putText(frame1, labels[0], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame2, labels[1], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side by side
        combined = np.hstack([frame1, frame2])
        
        # Add separator line
        cv2.line(combined, (width, 0), (width, height), (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(combined, f"Frame: {frame_count}", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(combined)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"✅ Comparison video saved: {output_path}")
    print(f"   Total frames: {frame_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Create side-by-side comparison video')
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original video')
    parser.add_argument('--tracked', type=str, required=True,
                       help='Path to tracked video')
    parser.add_argument('--output', type=str,
                       default='outputs/comparison.mp4',
                       help='Output video path')
    parser.add_argument('--label1', type=str, default='Original',
                       help='Label for first video')
    parser.add_argument('--label2', type=str, default='Tracked',
                       help='Label for second video')
    
    args = parser.parse_args()
    
    create_comparison_video(
        args.original,
        args.tracked,
        args.output,
        (args.label1, args.label2)
    )


if __name__ == '__main__':
    main()

