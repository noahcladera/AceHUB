# Tennis Video Processor

A complete end-to-end script for processing tennis videos from start to finish.

## Overview

This script automates the entire process of:
1. Taking videos and LLC (LosslessCut) files from the unprocessed_videos directory 
2. Extracting pose landmarks with MediaPipe
3. Normalizing the data
4. Processing segmentation based on the LLC files
5. Creating clips with skeleton overlays and 3D visualizations
6. Adding everything to the Strokes Library

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Plotly
- ffmpeg (for video conversion)

Install dependencies with:
```
pip install opencv-python mediapipe numpy pandas plotly
```

## Usage

### Preparing Videos and LLC Files

1. Place your tennis videos in the `unprocessed_videos` directory
2. Edit your videos with LosslessCut to mark the tennis strokes
3. Save the LLC file with the same name as the video in the `unprocessed_videos` directory
   - Example: `my_tennis_video.mp4` and `my_tennis_video.llc` or `my_tennis_video.mp4.llc`

### Running the Processor

Run the script with:
```
python src/personal/tennis_video_processor.py
```

This will process all video/LLC pairs in the `unprocessed_videos` directory.

### Options

- `--video-id ID`: Specify a starting video ID (useful if you want to control how videos are numbered)
- `--single`: Process only the first video found (useful for testing)

Example:
```
python src/personal/tennis_video_processor.py --video-id 10 --single
```

## Output

The script will:
1. Create numbered video directories in `videos/` and `data/processed/`
2. Generate clips, skeleton overlays, and 3D visualizations in the `Strokes_Library/` folder
3. Move processed videos and LLC files to `unprocessed_videos/archive/`

## Troubleshooting

- Make sure your LLC files contain valid stroke annotations
- If using LosslessCut, ensure you've saved the cut segments properly
- For videos in formats other than MP4, make sure ffmpeg is installed

## License

MIT 