# Test Videos for People Tracking

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
