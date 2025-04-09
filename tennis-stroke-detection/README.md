  _______ ______ _   _ _   _ _____  _____    _____ _______ _____   ____  _  ________ 
 |__   __|  ____| \ | | \ | |_   _|/ ____|  / ____|__   __|  __ \ / __ \| |/ /  ____|
    | |  | |__  |  \| |  \| | | | | (___   | (___    | |  | |__) | |  | | ' /| |__   
    | |  |  __| | . ` | . ` | | |  \___ \   \___ \   | |  |  _  /| |  | |  < |  __|  
    | |  | |____| |\  | |\  |_| |_ ____) |  ____) |  | |  | | \ \| |__| | . \| |____ 
    |_|  |______|_| \_|_| \_|_____|_____/  |_____/   |_|  |_|  \_\\____/|_|\_\______|
                                                                                      


# Tennis Stroke Detection

An AI-powered pipeline that processes tennis match videos to **detect**, **analyze**, and **visualize** tennis strokes. The system leverages **MediaPipe Pose** for extracting pose landmarks, **normalizes** and processes the data, and provides **deep learning** models (LSTM or Transformer) for stroke classification. It also offers utilities for **clip generation**, **visualization**, and **library creation**.

## Table of Contents

1. [Overview](#1-overview)  
2. [Features](#2-features)  
3. [Directory Structure](#3-directory-structure)  
4. [Installation](#4-installation)  
5. [Usage](#5-usage)  
6. [One-Click Processing](#6-one-click-processing)  
7. [Configuration](#7-configuration)  
8. [Model Architecture](#8-model-architecture)
9. [Contributing](#9-contributing)  
10. [License](#10-license)

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
- One-click processing script for simplified workflow.
- Visualization tools for skeleton overlays and analysis.

---

## 3. Directory Structure

```
tennis-stroke-detection/
├── README.md
├── Makefile
├── pyproject.toml
├── requirements-prod.txt
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── one-click.py
├── config/
│   ├── data_config.yaml
│   └── model_config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── data_pipeline.md
│   └── model_architecture.md
├── Final library/
└── src/
    ├── data/
    │   ├── acquisition.py
    │   ├── clip_extraction.py
    │   ├── clip_from_llc.py
    │   ├── create_clips_skeletons.py
    │   ├── dataset.py
    │   ├── feature_engineering.py
    │   ├── library_creation.py
    │   ├── library_time_normalization.py
    │   ├── normalization.py
    │   ├── pose_extraction.py
    │   ├── sequence_preparation.py
    │   └── show_skeleton.py
    ├── inference/
    │   ├── clip_generator.py
    │   └── predictor.py
    ├── models/
    │   ├── base.py
    │   ├── feature_extractors/
    │   │   └── pose_encoder.py
    │   └── sequence/
    │       ├── lstm.py
    │       └── transformer.py
    ├── personal/
    │   ├── create_skeleton_personal.py
    │   ├── normalization_personal.py
    │   └── pose_extraction_single.py
    ├── training/
    │   ├── callbacks.py
    │   ├── metrics.py
    │   └── trainer.py
    └── visualization/
        ├── pose_visualizer.py
        └── results_visualizer.py
```

---

## 4. Installation

1. **Clone** the repository:
```bash
git clone https://github.com/noahcladera/AceHUB.git
cd AceHUB/tennis-stroke-detection
```

2. **Virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

3. **Install** dependencies:
```bash
pip install -r requirements.txt     # for development
pip install -r requirements-prod.txt  # for production
```

4. **Environment variables**:
   - Copy `.env.example` to `.env` and configure as needed
   - Update paths in configuration files under `config/`

---

## 5. Usage

### Step-by-Step Workflow

The pipeline can be run step-by-step using individual scripts or via the Makefile:

1. **Data Acquisition**:
```bash
make download
# or
python src/data/acquisition.py
```

2. **Pose Extraction**:
```bash
make extract
# or
python src/data/pose_extraction.py
```

3. **Normalization**:
```bash
make normalize
# or
python src/data/normalization.py
```

4. **Feature Engineering**:
```bash
make feature-engineering
# or
python src/data/feature_engineering.py
```

5. **Model Training**:
```bash
make train
# or
python src/training/trainer.py
```

6. **Inference & Prediction**:
```bash
make predict
# or
python src/inference/predictor.py
```

7. **Clip Generation**:
```bash
make clip
# or
python src/inference/clip_generator.py
```

8. **Run Full Pipeline**:
```bash
make all
```

---

## 6. One-Click Processing

For simplified workflow, a comprehensive script is provided:

**One-Click Processing** (`one-click.py`):
- Combines all pipeline steps in a single script
- Downloads videos, extracts pose data, normalizes it, trains models, and generates clips
- Usage: `python one-click.py`

Configuration for the one-click process can be adjusted in the config files.

---

## 7. Configuration

Configuration files are located in the `config/` directory:

- `data_config.yaml`: Settings for data processing, paths, and parameters
- `model_config.yaml`: Model architecture, hyperparameters, and training settings

You can modify these files to customize the pipeline behavior.

---

## 8. Model Architecture

The system supports two main model architectures:

1. **LSTM Model**:
   - Long Short-Term Memory networks for sequence classification
   - Effective for temporal patterns in pose data
   - Configuration in `config/model_config.yaml`

2. **Transformer Model**:
   - Attention-based architecture for sequence modeling
   - Better at capturing long-range dependencies
   - Configuration in `config/model_config.yaml`

For more details, see `docs/model_architecture.md`.

---

## 9. Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 10. License

This project is licensed under the MIT License.
