#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check_Clips.py

This script does the following for each "video_x" folder in acehub/data:
1) Finds the labeled CSV (video_x_labeled.csv).
2) Finds the original video (video_x.mp4).
3) Reads the CSV, identifies stroke segments, and then clips those segments
   from the original video into a "video_x_clips" folder.
   
Configuration:
- FPS = 30 by default; adjust if your videos differ.
- The script expects each labeled CSV to have the format:
    - A 'frame_index' as the first column
    - A 'stroke_label' (0 or 1) as the last column
"""

import os
import csv
import ffmpeg

# ---------------
# CONFIGURATION
# ---------------
FPS = 30  # Frames per second of your videos

# Root folder structure: "acehub/data/video_1", etc.
ACEHUB_FOLDER = "acehub"
DATA_FOLDER = os.path.join(ACEHUB_FOLDER, "data")


def load_labels(csv_path):
    """
    Loads the labeled CSV and returns a list of tuples (frame_index, stroke_label).
    Assumes the first column is frame_index and the last column is stroke_label.
    """
    labels = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if exists
        for row in reader:
            # Convert frame index (first column) to integer and stroke_label (last column) to int
            frame_idx = int(float(row[0]))
            stroke_label = int(row[-1])
            labels.append((frame_idx, stroke_label))
    return labels

def get_stroke_segments(labels):
    """
    Given a list of (frame_index, stroke_label) tuples, returns a list of segments
    (start_frame, end_frame) where stroke_label == 1 continuously.
    """
    segments = []
    in_segment = False
    start_frame = None

    for frame_idx, label in labels:
        if label == 1 and not in_segment:
            # Start of a stroke segment
            in_segment = True
            start_frame = frame_idx
        elif label == 0 and in_segment:
            # End of a stroke segment
            in_segment = False
            end_frame = frame_idx - 1
            segments.append((start_frame, end_frame))

    # If CSV ends while still in a stroke segment
    if in_segment:
        segments.append((start_frame, labels[-1][0]))

    return segments

def clip_video_segments(video_path, segments, fps, output_folder):
    """
    Clips the original video into segments defined by (start_frame, end_frame) tuples.
    Saves each clip into the output_folder.
    """
    for i, (start_frame, end_frame) in enumerate(segments):
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        output_file = os.path.join(output_folder, f"stroke_{i+1}.mp4")

        print(f"[INFO] Clipping segment #{i+1}: start {start_time:.2f}s, end {end_time:.2f}s")
        try:
            (
                ffmpeg
                .input(video_path, ss=start_time, to=end_time)
                # Using codec="copy" for super-fast, lossless segmenting
                .output(output_file, codec="copy", loglevel="error", y=None)
                .run()
            )
        except Exception as e:
            print(f"[ERROR] Clipping segment #{i+1} failed: {e}")

def process_video_folder(video_folder):
    """
    1) Finds 'video_x_labeled.csv' in the folder
    2) Finds 'video_x.mp4'
    3) Reads the CSV, identifies stroke segments, and clips them.
    4) Saves clips into a new subfolder 'video_x_clips'
    """
    folder_name = os.path.basename(video_folder)  # e.g. "video_1"
    csv_filename = f"{folder_name}_labeled.csv"   # e.g. "video_1_labeled.csv"
    video_filename = f"{folder_name}.mp4"         # e.g. "video_1.mp4"

    labeled_csv_path = os.path.join(video_folder, csv_filename)
    original_video_path = os.path.join(video_folder, video_filename)

    # Check if both CSV and video exist
    if not os.path.isfile(labeled_csv_path):
        print(f"[SKIP] No labeled CSV found for {folder_name}: {labeled_csv_path}")
        return
    if not os.path.isfile(original_video_path):
        print(f"[SKIP] Original video file not found for {folder_name}: {original_video_path}")
        return

    # Load labels and identify stroke segments
    labels = load_labels(labeled_csv_path)
    if not labels:
        print(f"[INFO] No labels to process in {labeled_csv_path}")
        return

    segments = get_stroke_segments(labels)
    if not segments:
        print(f"[INFO] No stroke segments found in {folder_name} (all labels = 0?)")
        return

    # Create an output folder for clips, e.g. "video_1_clips"
    clips_folder_name = f"{folder_name}_clips"
    clips_folder_path = os.path.join(video_folder, clips_folder_name)
    os.makedirs(clips_folder_path, exist_ok=True)

    # Clip each stroke segment
    print(f"[INFO] Detected {len(segments)} stroke segments in {folder_name}")
    clip_video_segments(original_video_path, segments, FPS, clips_folder_path)
    print(f"[DONE] All segments clipped into: {clips_folder_path}")

def main():
    print(">>> Searching for labeled CSVs and videos in acehub/data/...")

    # Loop through all "video_x" folders under acehub/data
    for folder_name in os.listdir(DATA_FOLDER):
        folder_path = os.path.join(DATA_FOLDER, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Attempt to process that folder if it matches our structure
        process_video_folder(folder_path)

    print(">>> Done checking all folders.")

if __name__ == '__main__':
    main()
