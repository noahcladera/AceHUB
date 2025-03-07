# Data Pipeline

This document describes how data flows through the tennis stroke detection project.

1. **Raw Data Acquisition**  
   - Videos are downloaded from YouTube and stored in `data/raw/video_x/video_x.mp4`.
   - Additional metadata or label definitions can be downloaded if needed.

2. **Pose Extraction**  
   - Each video is processed frame by frame using MediaPipe Pose (script: `src/data/pose_extraction.py`).
   - Outputs a `_data.csv` file for each video in `data/interim/video_x`.

3. **Normalization**  
   - We apply spatial and temporal normalization (script: `src/data/normalization.py`) to produce `_normalized.csv` in `data/processed/video_x`.
   - This includes rotating the torso horizontally, scaling by torso length, smoothing, etc.

4. **Feature Engineering / Labeling**  
   - We merge manual stroke segmentation files (`.llc`) with normalized CSVs to generate `_labeled.csv` (script: `src/data/feature_engineering.py`).

5. **Training Data**  
   - The `_labeled.csv` files feed into model training (script: `src/training/trainer.py`).

6. **Inference / Clip Generation**  
   - Inference is run with `src/inference/predictor.py`.  
   - Clips for each detected stroke can be extracted with `src/inference/clip_generator.py`.

## Data Versions and Backup

Optionally, keep versions of your data in a separate storage or data versioning system (e.g., DVC, MLflow, Weights&Biases).