# Tennis Stroke Detection

An AI-powered pipeline that processes tennis match videos to **detect**, **analyze**, and **visualize** tennis strokes. The system leverages **MediaPipe Pose** for extracting pose landmarks, **normalizes** and processes the data, and provides **deep learning** models (LSTM or Transformer) for stroke classification. It also offers utilities for **clip generation**, **visualization**, and **library creation**.

## Table of Contents

1. Overview  
2. Features  
3. Directory Structure  
4. Installation  
5. Usage  
6. Extended Scripts & Final Library  
7. Configuration  
8. Contributing  
9. License

---

## 1. Overview

The pipeline includes:

- **Raw Video Acquisition** – Download/convert videos to uniform FPS.  
- **Pose Extraction** – Extract frame-by-frame landmarks via MediaPipe.  
- **Normalization** – Spatial & temporal normalization.  
- **Feature Engineering** – Merge stroke segments (LLC files) with normalized data to produce labeled CSV.  
- **Model Training** – LSTM or Transformer architectures to detect stroke sequences.  
- **Inference & Clipping** – Detect stroke segments in new videos, produce short MP4 clips.  
- **Visualization** – Render skeleton overlays, or create side-by-side comparisons for analysis.

---

## 2. Features

- Automated video processing (yt-dlp + FFmpeg).  
- Flexible pose extraction with MediaPipe.  
- Spatial and temporal normalization scripts.  
- Seamless label integration from LLC files.  
- Multiple model architectures (LSTM, Transformer).  
- Automated clip generation from labeled strokes.  
- Additional personal scripts for single-file or custom usage.

---

## 3. Directory Structure

noahcladera-acehub/ └── tennis-stroke-detection/ ├── README.md ├── Makefile ├── pyproject.toml ├── requirements-prod.txt ├── requirements.txt ├── setup.py ├── .env.example ├── .gitignore ├── abilities/ │ ├── data_validator.py │ ├── manual_labeling_ability.py │ └── model_converter.py # Currently empty placeholder ├── config/ │ ├── data_config.yaml │ ├── model_config.yaml │ └── pipeline_config.yaml ├── data/ │ ├── raw/ │ ├── interim/ │ └── processed/ ├── docs/ │ ├── data_pipeline.md │ └── model_architecture.md ├── Final library/ ├── models/ │ ├── base.py # Placeholder / mostly empty │ ├── ensemble/ │ │ └── stacking.py # Placeholder / mostly empty │ ├── feature_extractors/ │ │ └── pose_encoder.py │ └── sequence/ │ ├── lstm.py │ └── transformer.py └── src/ ├── data/ │ ├── acquisition.py │ ├── clip_extraction.py │ ├── clip_from_llc.py │ ├── create_clips_skeletons.py │ ├── dataset.py # Currently empty │ ├── feature_engineering.py │ ├── library_creation.py │ ├── library_time_normalization.py │ ├── normalization.py │ ├── pose_extraction.py │ ├── sequence_preparation.py │ └── show_skeleton.py ├── inference/ │ ├── clip_generator.py │ └── predictor.py ├── personal/ │ ├── create_skeleton_personal.py │ ├── normalization_personal.py │ └── pose_extraction_single.py ├── training/ │ ├── callbacks.py │ ├── metrics.py │ └── trainer.py └── visualization/ ├── pose_visualizer.py └── results_visualizer.py


---

## 4. Installation

1. **Clone** the repository:
```bash
   git clone <repository_url>
   cd noahcladera-acehub/tennis-stroke-detection
```
2. **Virtual environment** (recommended):
```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
```
3. **Install** dependencies:
```bash
    pip install -r requirements.txt     # for development
    pip install -r requirements-prod.txt  # for production
```
