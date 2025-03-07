#!/usr/bin/env python3
"""
setup_folder_structure.py

Creates a clean folder structure for the AceHub project.

Structure:
acehub/
  ├── scripts/
  ├── data/
  │    ├── video_1/
  │    │    ├── video_1.mp4
  │    │    ├── video_1_data.csv
  │    │    ├── video_1_normalized.csv
  │    │    ├── video_1.llc
  │    │    └── video_1_labeled.csv
  │    ├── video_2/
  │    │    └── ...
  │    └── ...
  ├── results/
  │    ├── video_1_strokes/
  │    ├── video_2_strokes/
  │    └── ...
  └── README.md (optional)
"""

import os

# You can list your videos here if you already know them:
VIDEO_LIST = [
    "video_1",
    "video_2",
    # Add more as needed
]

def create_folder_structure(base_folder="acehub"):
    """
    Creates the recommended folder structure inside `base_folder`.
    """
    # 1) Create main directories
    scripts_path = os.path.join(base_folder, "scripts")
    data_path = os.path.join(base_folder, "data")
    results_path = os.path.join(base_folder, "results")

    os.makedirs(scripts_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # 2) For each video, create a subfolder in data/
    for vid in VIDEO_LIST:
        vid_folder = os.path.join(data_path, vid)
        os.makedirs(vid_folder, exist_ok=True)

        # (Optional) Also create a strokes folder in results if you want
        # to store final stroke clips for that video
        stroke_folder = os.path.join(results_path, f"{vid}_strokes")
        os.makedirs(stroke_folder, exist_ok=True)

    # 3) (Optional) Create a placeholder README in the base folder
    readme_path = os.path.join(base_folder, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# AceHub Project\n\nFolder structure for the project.\n")

    print(f"Folder structure created under '{base_folder}'")


if __name__ == "__main__":
    create_folder_structure()
