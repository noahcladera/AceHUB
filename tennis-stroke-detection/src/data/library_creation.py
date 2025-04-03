#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_clips_flat.py

Collects all clip files (MP4, CSV) from data/processed/video_x/video_x_clips
and places them into a single folder: "Final library".
No subfolders are used. To avoid collisions, each file is renamed:
  "video_3_clip_2.mp4"
  "video_3_clip_2.csv"
...where "video_3" is the parent folder, and "clip_2" is the original name.

Result:
  Final library/
    video_1_clip_1.mp4
    video_1_clip_1.csv
    video_1_clip_2.mp4
    video_1_clip_2.csv
    video_2_clip_1.mp4
    ...
    video_3_clip_1.mp4
    video_3_clip_1.csv
    ...
"""

import os
import shutil

PROCESSED_DIR = os.path.join("data", "processed")
FINAL_LIBRARY = "Final library"  # The single folder where all clips will be stored

def main():
    # Create the final library folder if it doesn't exist
    os.makedirs(FINAL_LIBRARY, exist_ok=True)

    # Iterate over every subfolder in data/processed
    for folder_name in os.listdir(PROCESSED_DIR):
        folder_path = os.path.join(PROCESSED_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # We look for subfolders named video_x
        if not folder_name.startswith("video_"):
            continue

        # The clip folder is data/processed/video_x/video_x_clips
        clips_subfolder = os.path.join(folder_path, f"{folder_name}_clips")
        if not os.path.isdir(clips_subfolder):
            continue

        # For each file in the clips folder (e.g. clip_1.mp4, clip_1.csv, etc.)
        for filename in os.listdir(clips_subfolder):
            src_file = os.path.join(clips_subfolder, filename)
            if not os.path.isfile(src_file):
                continue

            # We'll rename "clip_1.mp4" to "video_3_clip_1.mp4" if folder_name is "video_3"
            # So the result is "video_3_clip_1.mp4"
            dest_filename = f"{folder_name}_{filename}"  # e.g. "video_3_clip_1.mp4"
            dest_path = os.path.join(FINAL_LIBRARY, dest_filename)

            # Copy the file
            shutil.copy2(src_file, dest_path)
            print(f"[COPY] {src_file} -> {dest_path}")

    print(f"[DONE] All clips consolidated into: {FINAL_LIBRARY}")

if __name__ == "__main__":
    main()
