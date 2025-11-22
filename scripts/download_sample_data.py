"""Script to download sample test videos for people tracking"""
import os
import sys
import urllib.request
from pathlib import Path

def download_sample_video():
    """Download a sample test video"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Sample Video Download Options")
    print("=" * 60)
    print()
    print("Option 1: Use your webcam (no download needed)")
    print("  python -m src.inference.main --input 0 --display")
    print()
    print("Option 2: Download sample videos from public sources")
    print()
    print("Recommended sources:")
    print("1. MOT Challenge Dataset:")
    print("   https://motchallenge.net/data/")
    print("   - MOT17: https://motchallenge.net/data/MOT17/")
    print("   - MOT20: https://motchallenge.net/data/MOT20/")
    print()
    print("2. PETS Dataset:")
    print("   http://www.cvg.reading.ac.uk/PETS2009/a.html")
    print()
    print("3. Create your own test video:")
    print("   - Record with your phone/camera")
    print("   - Use any video with people in it")
    print("   - Place it in: data/raw/")
    print()
    print("4. YouTube (for testing only, respect copyright):")
    print("   - Download using yt-dlp: pip install yt-dlp")
    print("   - yt-dlp <youtube_url> -o data/raw/test_video.mp4")
    print()
    
    # Try to download a small test video from a public source
    print("Attempting to download a small test video...")
    
    # Use a small public test video (example - you may need to update URLs)
    test_videos = [
        {
            "name": "sample_people.mp4",
            "url": None,  # Add a direct download URL if available
            "description": "Sample people tracking video"
        }
    ]
    
    print("\nNote: For legal and practical reasons, we recommend:")
    print("1. Using your webcam for immediate testing")
    print("2. Recording your own test videos")
    print("3. Downloading from official datasets (MOT Challenge, PETS)")
    print()
    print("To test with webcam right now, run:")
    print("  python -m src.inference.main --input 0 --display")


def create_test_video_info():
    """Create a README with video download instructions"""
    readme_path = Path("data/raw/README.md")
    readme_content = """# Test Videos for People Tracking

## Quick Start - Use Webcam
No download needed! Just run:
```bash
python -m src.inference.main --input 0 --display
```

## Download Sample Videos

### Option 1: MOT Challenge Dataset (Recommended)
1. Visit: https://motchallenge.net/data/
2. Download MOT17 or MOT20 sequences
3. Extract videos to this directory

### Option 2: Record Your Own
- Use your phone/camera to record a video with people
- Save it as MP4, AVI, or MOV format
- Place it in this directory

### Option 3: Public Test Videos
You can use any video file with people in it for testing.

## Video Format Requirements
- Formats: MP4, AVI, MOV, MKV
- Recommended: MP4 (H.264 codec)
- Resolution: Any (system will handle it)

## Example Usage
Once you have a video file (e.g., `test_video.mp4`):
```bash
python -m src.inference.main --input data/raw/test_video.mp4 --display
```
"""
    
    readme_path.write_text(readme_content)
    print(f"Created README at: {readme_path}")


def main():
    """Main function"""
    create_test_video_info()
    download_sample_video()


if __name__ == "__main__":
    main()

