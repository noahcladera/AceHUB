#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_engineering.py

Iterates over each video folder in "tennis-stroke-detection/data/processed".
Reads the normalized pose CSV file (e.g., "video_1_normalized.csv") and the
manual segmentation file (LLC file, e.g. "video_1.llc"), converts the cut segments
from seconds to frame indices (using FPS=30), and then writes a new CSV file
with an added column "stroke_label" (1 if the frame is within any stroke, 0 otherwise).
"""

import os
import csv
import json
import math
import re

#############################################
# CONFIGURATION SETTINGS
#############################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")

FPS = 30  # Frames per second (used to convert seconds to frame indices)

#############################################
# HELPER FUNCTIONS
#############################################
def convert_llc_to_valid_json(text):
    """
    Converts LLC file text to valid JSON:
      - Wraps unquoted keys with double quotes.
      - Replaces single quotes with double quotes.
      - Removes trailing commas before '}' or ']'.
    """
    # Wrap unquoted keys (after '{' or ',' preceding a colon) with double quotes.
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', text)
    # Replace single quotes with double quotes.
    text = text.replace("'", '"')
    # Remove trailing commas before '}' or ']'
    text = re.sub(r',\s*(\}|])', r'\1', text)
    return text

def load_segments_from_llc(llc_path):
    """
    Loads the manual cut segments from an LLC file
    (which may not be valid JSON by default).
    Returns a list of (start_frame, end_frame) tuples (integers).
    """
    with open(llc_path, 'r') as f:
        raw_text = f.read()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        valid_text = convert_llc_to_valid_json(raw_text)
        data = json.loads(valid_text)

    segments = []
    for seg in data["cutSegments"]:
        start_frame = int(math.floor(seg["start"] * FPS))
        end_frame   = int(math.floor(seg["end"] * FPS))
        segments.append((start_frame, end_frame))
    return segments

def create_frame_labels(csv_path, llc_path, output_csv):
    """
    Reads the normalized pose CSV and the LLC file, and writes a new CSV
    with an added "stroke_label" column (0 or 1) indicating whether the frame
    is within any of the segments in the LLC.
    """
    segments = load_segments_from_llc(llc_path)

    def is_frame_in_stroke(frame_idx):
        for (start, end) in segments:
            if start <= frame_idx <= end:
                return True
        return False

    with open(csv_path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        new_header = header + ["stroke_label"]
        rows = []
        for row in reader:
            # The first column is frame_index (which might be float in some CSVs)
            frame_idx = int(float(row[0]))
            label = 1 if is_frame_in_stroke(frame_idx) else 0
            rows.append(row + [str(label)])

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(rows)

    print(f"[INFO] Created labeled CSV -> {output_csv}")

#############################################
# MAIN EXECUTION
#############################################
def main():
    # Iterate over each subfolder in BASE_DATA_DIR
    for folder in os.listdir(BASE_DATA_DIR):
        folder_path = os.path.join(BASE_DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # e.g., "video_1"
        video_basename = folder
        input_csv = os.path.join(folder_path, f"{video_basename}_normalized.csv")
        llc_file = os.path.join(folder_path, f"{video_basename}.llc")
        output_csv = os.path.join(folder_path, f"{video_basename}_labeled.csv")

        if not os.path.exists(input_csv):
            print(f"[WARNING] Normalized CSV not found in {folder_path}. Skipping.")
            continue
        if not os.path.exists(llc_file):
            print(f"[WARNING] LLC file not found in {folder_path}. Skipping.")
            continue

        print(f"[INFO] Processing folder: {folder_path}")
        create_frame_labels(input_csv, llc_file, output_csv)

if __name__ == "__main__":
    main()