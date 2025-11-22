"""Script to process MOT20 sequences for people tracking"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.mot_utils import list_mot_sequences, mot_sequence_to_video, get_mot_sequences


def main():
    parser = argparse.ArgumentParser(description="Process MOT20 sequences")
    parser.add_argument("--list", action="store_true", help="List available sequences")
    parser.add_argument("--convert", type=str, help="Convert sequence to video (sequence name)")
    parser.add_argument("--convert-all", action="store_true", help="Convert all sequences to videos")
    parser.add_argument("--mot-dir", type=str, default="data/raw/MOT20/MOT20", 
                       help="Path to MOT dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/mot20_videos",
                       help="Output directory for converted videos")
    
    args = parser.parse_args()
    
    # List sequences
    if args.list:
        list_mot_sequences(args.mot_dir)
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert single sequence
    if args.convert:
        sequences = get_mot_sequences(args.mot_dir)
        seq = next((s for s in sequences if s['name'] == args.convert), None)
        
        if not seq:
            print(f"Error: Sequence '{args.convert}' not found")
            print("\nAvailable sequences:")
            list_mot_sequences(args.mot_dir)
            return
        
        output_path = output_dir / f"{seq['name']}.mp4"
        print(f"Converting {seq['name']} to video...")
        mot_sequence_to_video(seq['path'], str(output_path))
        print(f"\n✓ Video saved to: {output_path}")
        print(f"\nYou can now process it with:")
        print(f"  python -m src.inference.main --input {output_path} --display")
    
    # Convert all sequences
    elif args.convert_all:
        sequences = get_mot_sequences(args.mot_dir)
        
        if not sequences:
            print(f"No sequences found in {args.mot_dir}")
            return
        
        print(f"Converting {len(sequences)} sequences to videos...\n")
        
        for seq in sequences:
            output_path = output_dir / f"{seq['name']}.mp4"
            print(f"\n{'='*60}")
            print(f"Processing: {seq['name']} ({seq['split']})")
            print(f"{'='*60}")
            
            if mot_sequence_to_video(seq['path'], str(output_path)):
                print(f"✓ Success: {output_path}")
            else:
                print(f"✗ Failed: {seq['name']}")
        
        print(f"\n{'='*60}")
        print(f"All videos saved to: {output_dir}")
        print(f"{'='*60}")
    
    else:
        print("MOT20 Sequence Processor")
        print("\nUsage:")
        print("  List sequences:     python scripts/process_mot_sequences.py --list")
        print("  Convert one:        python scripts/process_mot_sequences.py --convert MOT20-01")
        print("  Convert all:        python scripts/process_mot_sequences.py --convert-all")
        print("\nOr use sequences directly with image sequence support:")
        print("  python -m src.inference.main --input data/raw/MOT20/MOT20/train/MOT20-01/img1 --display")


if __name__ == "__main__":
    main()

