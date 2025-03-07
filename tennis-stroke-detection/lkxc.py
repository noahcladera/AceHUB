#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
update_pipeline_paths.py

Applies the recommended path fixes so your pipeline follows a:
  raw -> interim -> processed
data flow, and removes references to "acehub/data" in clip_generator.py.

Usage:
    python update_pipeline_paths.py
"""

import os

# Adjust these paths to point to where your scripts actually reside
POSE_EXTRACTION_PATH = "tennis-stroke-detection/src/data/pose_extraction.py"
NORMALIZATION_PATH = "tennis-stroke-detection/src/data/normalization.py"
CLIP_GENERATOR_PATH = "tennis-stroke-detection/src/inference/clip_generator.py"

def update_pose_extraction():
    """
    1) Change BASE_DATA_DIR in pose_extraction.py from data/raw to data/interim
    2) Possibly adjust any textual references or docstrings mentioning 'raw' to say 'interim'.
    """
    if not os.path.isfile(POSE_EXTRACTION_PATH):
        print(f"[SKIP] {POSE_EXTRACTION_PATH} not found.")
        return

    with open(POSE_EXTRACTION_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Example: BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")
        if "BASE_DATA_DIR" in line and "data" in line and "raw" in line:
            # Replace raw with interim
            line = line.replace('"data", "raw"', '"data", "interim"')
        new_lines.append(line)

    with open(POSE_EXTRACTION_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"[DONE] Updated pose_extraction.py to store _data.csv in data/interim.")


def update_normalization():
    """
    1) normalization.py currently reads from data/interim (that’s fine).
    2) Make sure it writes _normalized.csv to data/processed.
    """
    if not os.path.isfile(NORMALIZATION_PATH):
        print(f"[SKIP] {NORMALIZATION_PATH} not found.")
        return

    with open(NORMALIZATION_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Example: BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "interim")
        # Keep reading from interim! That’s good for the input.
        # We only want to ensure the output is in data/processed. In the code, we see:
        #    output_csv = data_csv.replace("_data.csv", "_normalized.csv")
        # which writes in the same folder. We'll add logic to place it in processed.
        if "output_csv = data_csv.replace(" in line:
            # Insert logic: replace path from 'interim' to 'processed'
            # E.g. something like: output_csv = data_csv.replace("data/interim", "data/processed")
            # We'll do a naive approach:
            # Because data_csv is something like "/data/interim/video_1/video_1_data.csv",
            # let's just do a replace on "data/interim" -> "data/processed".

            line = line.replace('data_csv.replace("_data.csv", "_normalized.csv")',
                                'data_csv.replace("data/interim", "data/processed").replace("_data.csv", "_normalized.csv")')

        new_lines.append(line)

    with open(NORMALIZATION_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"[DONE] Updated normalization.py to output _normalized.csv in data/processed.")


def update_clip_generator():
    """
    1) Remove references to 'acehub' folder.
    2) Change DATA_FOLDER from os.path.join(ACEHUB_FOLDER, 'data') to
       'tennis-stroke-detection/data' or similar.
    """
    if not os.path.isfile(CLIP_GENERATOR_PATH):
        print(f"[SKIP] {CLIP_GENERATOR_PATH} not found.")
        return

    with open(CLIP_GENERATOR_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # e.g. ACEHUB_FOLDER = "acehub"
        if 'ACEHUB_FOLDER = "acehub"' in line:
            line = line.replace('ACEHUB_FOLDER = "acehub"',
                                'ACEHUB_FOLDER = "tennis-stroke-detection"')

        # e.g. DATA_FOLDER = os.path.join(ACEHUB_FOLDER, "data")
        if 'DATA_FOLDER = os.path.join(ACEHUB_FOLDER, "data")' in line:
            line = line.replace('DATA_FOLDER = os.path.join(ACEHUB_FOLDER, "data")',
                                'DATA_FOLDER = os.path.join("tennis-stroke-detection", "data")')

        new_lines.append(line)

    with open(CLIP_GENERATOR_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"[DONE] Updated clip_generator.py to reference tennis-stroke-detection/data.")


def main():
    print("[INFO] Starting pipeline path fixes...")
    update_pose_extraction()
    update_normalization()
    update_clip_generator()
    print("[INFO] All done.")


if __name__ == "__main__":
    main()