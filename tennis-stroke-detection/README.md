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
│   ├── raw/                     # Raw downloaded videos
│   │   └── video_1.mp4
│   │   └── video_2.mp4
│   ├── processed/               # Processed data (pose, normalized)
│   │   └── video_1/
│   │       └── video_1_data.csv
│   │       └── video_1_normalized.csv
│   │       └── video_1.llc
│   └── Strokes_Library/         # Final organized stroke clips
│       └── stroke_1/            # Each stroke in its own folder
│           └── stroke.csv       # Raw pose data for this stroke
│           └── stroke_norm.csv  # Normalized pose data
│           └── stroke_clip.mp4  # Video clip
│           └── stroke_skeleton.mp4 # Skeleton overlay
│           └── stroke_overlay.mp4  # Pose overlay
├── scripts/                     # Utility scripts
│   └── process_video.py         # Command-line interface
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

The pipeline can be run using the command-line interface:

### Process a Video

To process a video (download, extract pose, normalize):

```bash
python scripts/process_video.py process <youtube_url> --output video_1
```

Options:
- `--output` or `-o`: Name for the output video (default: video_1)
- `--fps`: Target FPS (default: 30)

### Process Stroke Segments

After manually segmenting the video with LosslessCut:

```bash
python scripts/process_video.py strokes video_1
```

Options:
- `--force` or `-f`: Force reprocessing of existing files

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
