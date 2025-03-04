#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5Create_Frame_Labels.py

This script reads a normalized pose CSV and a manual segmentation file (LLC file)
that lists stroke segments by seconds. It converts the seconds to frame numbers
(using a specified FPS) and then, for each frame in the CSV, adds a new column "stroke_label"
which is 1 if that frame is within any stroke segment and 0 otherwise.
"""

import os
import csv
import json
import math
import re

#############################################
# CONFIGURATION SETTINGS
#############################################
VIDEO_BASENAME = "video_1"  # e.g., "video_1"
MEDIA_FOLDER = os.path.join("Test_media", "test_videos")

# Input CSV: for example, "video_1_normalized.csv"
INPUT_CSV = os.path.join(MEDIA_FOLDER, f"{VIDEO_BASENAME}_normalized.csv")

# The LLC file that contains manual cut segments.
# Note: In your folder the LLC file is named, for example, "video_1-proj.llc"
LLC_FILE = os.path.join(MEDIA_FOLDER, f"{VIDEO_BASENAME}.llc")

# Output CSV: a new CSV file with an added column "stroke_label"
OUTPUT_CSV = os.path.join(MEDIA_FOLDER, f"{VIDEO_BASENAME}_labeled.csv")

# Video FPS (frames per second)
FPS = 30

#############################################
# FUNCTIONS
#############################################
def convert_llc_to_valid_json(text):
    """
    Converts LLC file text to valid JSON:
      - Wraps unquoted keys with double quotes.
      - Replaces single quotes with double quotes.
      - Removes trailing commas.
    """
    # Wrap unquoted keys (after '{' or ',' preceding a colon) with double quotes.
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', text)
    # Replace single quotes with double quotes.
    text = text.replace("'", '"')
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*(\}|])', r'\1', text)
    return text


def load_segments_from_llc(llc_path):
    """
    Loads the manual cut segments from an LLC file.
    Converts the LLC file to valid JSON if necessary.
    Returns a list of (start_frame, end_frame) tuples (as integers),
    where the start and end values (originally in seconds) are converted to frames.
    """
    with open(llc_path, 'r') as f:
        raw_text = f.read()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        valid_text = convert_llc_to_valid_json(raw_text)
        data = json.loads(valid_text)
    segments = []
    # Multiply the seconds by FPS to convert to frame numbers.
    for seg in data["cutSegments"]:
        start_frame = int(math.floor(seg["start"] * FPS))
        end_frame   = int(math.floor(seg["end"] * FPS))
        segments.append((start_frame, end_frame))
    return segments


def create_frame_labels(csv_path, llc_path, output_csv):
    """
    Reads the normalized pose CSV and the LLC file (converted to JSON),
    and writes a new CSV with an added "stroke_label" column (0 or 1).
    """
    # 1) Load all segments from the LLC file.
    segments = load_segments_from_llc(llc_path)

    # 2) Helper function to check if a frame is in any stroke segment.
    def is_frame_in_stroke(frame_idx):
        for (start, end) in segments:
            if start <= frame_idx <= end:
                return True
        return False

    # 3) Read the CSV and parse each row.
    with open(csv_path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        new_header = header + ["stroke_label"]

        rows = []
        for row in reader:
            # Assume the first column is frame_index.
            frame_idx = int(float(row[0]))
            label = 1 if is_frame_in_stroke(frame_idx) else 0
            rows.append(row + [str(label)])

    # 4) Write out the new CSV.
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(rows)

    print(f"[INFO] Created labeled CSV -> {output_csv}")


#############################################
# MAIN EXECUTION
#############################################
if __name__ == "__main__":
    create_frame_labels(INPUT_CSV, LLC_FILE, OUTPUT_CSV)
