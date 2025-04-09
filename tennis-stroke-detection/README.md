# Tennis Stroke Detection

A pipeline for processing tennis videos, extracting pose data, and creating segmented stroke clips.

## Overview

This project processes tennis videos through a two-stage pipeline:

### Video Processing Pipeline
1. **Video Acquisition**: Download videos from YouTube and convert to 30 FPS
2. **Pose Extraction**: Extract pose landmarks using MediaPipe
3. **Normalization**: Apply spatial normalization to pose data

### Manual Segmentation
After the video processing stage, use LosslessCut to manually segment tennis strokes and export LLC files.

### Stroke Segmentation Pipeline
1. **Feature Engineering**: Add stroke labels to frames based on LLC files
2. **Clip Generation**: Create video clips from labeled segments
3. **Clip Collection**: Collect all clips into a "Strokes_Library" folder
4. **Time Normalization**: Normalize clip timelines to a fixed number of frames

## Project Structure

```
tennis-stroke-detection/
├── src/
│   ├── video_processing.py      # Video processing pipeline
│   └── stroke_segmentation.py   # Stroke segmentation pipeline
├── config/
│   ├── video_config.yaml        # Video processing settings
│   └── segmentation_config.yaml # Stroke segmentation settings
├── videos/                      # All video-related data
│   └── video_X/                 # Each video in its own folder
│       ├── video_X.mp4          # Original video
│       ├── video_X_data.csv     # Raw pose data 
│       ├── video_X_normalized.csv # Normalized pose data
│       ├── video_X.llc          # Manual segmentation file
│       ├── status.txt           # Processing status
│       └── video_X_clips/       # Generated stroke clips
│           ├── stroke_1.mp4
│           ├── stroke_1.csv
│           └── ...
├── Strokes_Library/             # Final organized stroke clips
│   └── stroke_Y/                # Each stroke in its own folder
│       ├── stroke.csv           # Raw pose data for this stroke
│       ├── stroke_norm.csv      # Normalized pose data
│       ├── stroke_clip.mp4      # Video clip
│       ├── stroke_skeleton.mp4  # Skeleton overlay
│       ├── stroke_overlay.mp4   # Pose overlay
│       └── source_info.txt      # Reference to source video
├── scripts/                     # Utility scripts
│   ├── process_video.py         # Command-line interface
│   ├── reorganize_videos.py     # Organize videos into new structure
│   └── find_and_migrate_videos.py # Migrate existing videos
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/noahcladera/tennis-stroke-detection.git
cd tennis-stroke-detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Migrating Existing Videos

If you have existing videos in the `data/` directory structure, you can migrate them to the new structure:

```bash
python find_and_migrate_videos.py
```

### Adding New Videos

To add a new video to the system:

1. Create a folder for it in the `videos/` directory:

```bash
mkdir -p videos/video_X
```

2. Copy your video file, normalized CSV, and other files into this folder.

3. Create an LLC file using LosslessCut to manually segment the strokes.

### Processing Videos for Stroke Detection

After setting up videos in the `videos/` directory:

```bash
python src/stroke_segmentation.py
```

This script will:
1. Process each video that has the required files
2. Create stroke clips from the LLC segments
3. Add the clips to the Strokes_Library in a standardized format

## Configuration

The pipeline is configured via YAML files in the `config/` directory:

- `video_config.yaml`: Configure video processing and pose extraction settings
- `segmentation_config.yaml`: Configure stroke segmentation and clip generation settings

## Dependencies

- Python 3.8+
- FFmpeg
- MediaPipe
- OpenCV
- NumPy
- LosslessCut (for manual segmentation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation
- [PyTorch](https://pytorch.org/) for deep learning models
- [OpenCV](https://opencv.org/) for video processing
- [FFmpeg](https://ffmpeg.org/) for video manipulation
