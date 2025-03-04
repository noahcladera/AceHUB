#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
6Check_Clips.py

This script reads the labeled CSV file (which contains a "stroke_label" column),
identifies contiguous segments where the label is 1, and clips the corresponding
parts from the original video. It converts frame indices to seconds using a specified FPS.
"""

import os
import csv
import math
import ffmpeg

# CONFIGURATION
FPS = 30  # Frames per second of the video

# Update these paths according to your folder structure
LABELED_CSV = os.path.join("Test_media", "test_videos", "video_1_labeled.csv")
ORIGINAL_VIDEO = os.path.join("Test_media", "test_videos", "video_1.mp4")
OUTPUT_FOLDER = os.path.join("Test_media", "test_videos", "labeled_strokes")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_labels(csv_path):
    """
    Loads the labeled CSV and returns a list of tuples (frame_index, stroke_label).
    Assumes the first column is frame_index and the last column is stroke_label.
    """
    labels = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
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
            # End of a stroke segment; take the last frame that had label==1
            in_segment = False
            end_frame = frame_idx - 1
            segments.append((start_frame, end_frame))
    # If the file ends while still in a stroke segment
    if in_segment:
        segments.append((start_frame, labels[-1][0]))
    return segments

def clip_video_segments(video_path, segments, fps, output_folder):
    """
    Clips the original video into segments defined by the provided (start_frame, end_frame) tuples.
    Saves each clip to the output_folder.
    """
    for i, (start_frame, end_frame) in enumerate(segments):
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        output_file = os.path.join(output_folder, f"stroke_{i+1}.mp4")
        print(f"Clipping segment {i+1}: start {start_time:.2f}s, end {end_time:.2f}s, duration {duration:.2f}s")
        try:
            (
                ffmpeg
                .input(video_path, ss=start_time, to=end_time)
                .output(output_file, codec="copy", loglevel="error", y=None)
                .run()
            )
        except Exception as e:
            print(f"Error clipping segment {i+1}: {e}")

def main():
    print("[INFO] Loading labeled CSV...")
    labels = load_labels(LABELED_CSV)
    print(f"[INFO] Loaded {len(labels)} frames.")

    segments = get_stroke_segments(labels)
    print(f"[INFO] Detected {len(segments)} stroke segments:")
    for seg in segments:
        print(f"    Frames {seg[0]} to {seg[1]}")

    print("[INFO] Clipping video segments...")
    clip_video_segments(ORIGINAL_VIDEO, segments, FPS, OUTPUT_FOLDER)
    print("[INFO] All segments clipped.")

if __name__ == '__main__':
    main()
