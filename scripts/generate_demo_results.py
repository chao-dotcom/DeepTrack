"""Generate demo results for presentation"""
import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    """Generate demo results"""
    print("="*70)
    print("Generating Demo Results for People Tracking System")
    print("="*70)
    
    # Create output directory
    output_dir = Path("data/processed/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nThis script will generate:")
    print("1. Processed video with tracking visualization")
    print("2. Results JSON with statistics")
    print("3. Summary report")
    
    # Check if model exists
    model_path = Path("models/checkpoints/yolov8n.pt")
    if not model_path.exists():
        print("\n⚠ Model not found. Downloading...")
        os.system("python scripts/download_models.py")
    
    # Find available MOT20 sequences
    mot_train = Path("data/raw/MOT20/MOT20/train")
    sequences = []
    
    if mot_train.exists():
        for seq_dir in mot_train.iterdir():
            img_dir = seq_dir / "img1"
            if img_dir.exists():
                img_count = len(list(img_dir.glob("*.jpg")))
                sequences.append({
                    'name': seq_dir.name,
                    'path': str(img_dir),
                    'frames': img_count
                })
    
    if not sequences:
        print("\n✗ No MOT20 sequences found!")
        print("Please ensure MOT20 dataset is in data/raw/MOT20/MOT20/")
        return
    
    # Sort by frame count (use smaller sequences for demo)
    sequences.sort(key=lambda x: x['frames'])
    
    print(f"\nFound {len(sequences)} sequences:")
    for i, seq in enumerate(sequences[:5], 1):
        print(f"  {i}. {seq['name']}: {seq['frames']} frames")
    
    # Process first sequence (smallest)
    selected = sequences[0]
    print(f"\n▶ Processing: {selected['name']} ({selected['frames']} frames)")
    
    output_video = output_dir / f"{selected['name']}_demo_{timestamp}.mp4"
    output_json = output_dir / f"{selected['name']}_results_{timestamp}.json"
    output_report = output_dir / f"demo_report_{timestamp}.md"
    
    # Run inference
    cmd = f'python -m src.inference.main --input "{selected["path"]}" --output "{output_video}" --model models/checkpoints/yolov8n.pt'
    
    print(f"\nRunning inference...")
    print(f"Command: {cmd}\n")
    
    result = os.system(cmd)
    
    if result == 0 and output_video.exists():
        print(f"\n✓ Success! Generated files:")
        print(f"  📹 Video: {output_video}")
        print(f"  📊 Results: {output_json}")
        
        # Generate markdown report
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(f"# People Tracking System - Demo Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Input Sequence\n")
            f.write(f"- **Name:** {selected['name']}\n")
            f.write(f"- **Frames:** {selected['frames']}\n")
            f.write(f"- **Path:** {selected['path']}\n\n")
            f.write(f"## Output Files\n")
            f.write(f"- **Video:** `{output_video.name}`\n")
            f.write(f"- **Results JSON:** `{output_json.name}`\n\n")
            f.write(f"## How to View Results\n\n")
            f.write(f"1. **Video Output:** Open `{output_video.name}` in any video player\n")
            f.write(f"2. **Statistics:** Check `{output_json.name}` for detailed metrics\n")
            f.write(f"3. **System Features:**\n")
            f.write(f"   - Real-time people detection using YOLOv8\n")
            f.write(f"   - Multi-object tracking with unique IDs\n")
            f.write(f"   - Visual bounding boxes and track labels\n\n")
            f.write(f"## System Capabilities Demonstrated\n\n")
            f.write(f"- ✅ Person detection in video sequences\n")
            f.write(f"- ✅ Multi-object tracking with ID assignment\n")
            f.write(f"- ✅ MOT20 dataset processing\n")
            f.write(f"- ✅ Video output generation\n")
            f.write(f"- ✅ Results statistics and reporting\n")
        
        print(f"  📄 Report: {output_report}")
        print(f"\n{'='*70}")
        print("Demo results generated successfully!")
        print(f"{'='*70}")
        print(f"\nAll files saved to: {output_dir}")
        print(f"\nYou can now show:")
        print(f"  - Video with tracking: {output_video.name}")
        print(f"  - Results data: {output_json.name}")
        print(f"  - Demo report: {output_report.name}")
    else:
        print("\n✗ Processing failed. Check errors above.")

if __name__ == "__main__":
    main()

