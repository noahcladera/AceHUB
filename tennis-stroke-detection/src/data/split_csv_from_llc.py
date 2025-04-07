#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_csv_from_llc.py

Steps:
1) Under data/processed/video_x, find video_x.llc and video_x_normalized.csv.
2) Read .llc to get cutSegments with start/end times in seconds.
3) Convert each segment to frame indices using a known FPS (e.g. 30).
4) Slice the normalized CSV rows whose frame_index is in [start_frame, end_frame].
5) Write each slice to clip_1.csv, clip_2.csv, etc. in the same folder as the video clips:
     data/processed/video_x/video_x_clips/

So each clip_#.csv is side-by-side with clip_#.mp4.
"""

import os
import re
import json
import math
import csv

FPS = 30  # Must match your normalization FPS
PROCESSED_DIR = os.path.join("data", "processed")  # e.g., "data/processed/video_3/"

def convert_llc_to_valid_json(raw_text):
    """
    Converts typical .llc text (with single quotes, unquoted keys, trailing commas)
    into valid JSON so we can parse it with `json.loads()`.
    """
    # 1) Wrap unquoted keys that appear before a colon
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', raw_text)
    # 2) Replace single quotes with double quotes
    text = text.replace("'", '"')
    # 3) Remove trailing commas before } or ]
    text = re.sub(r',\s*(\}|])', r'\1', text)
    return text

def load_llc_segments(llc_path):
    """
    Reads the .llc file and returns a list of dicts like:
      [
        {"start": 0.0, "end": 1.55, "name": ""},
        ...
      ]
    """
    with open(llc_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    valid_json = convert_llc_to_valid_json(raw)
    data = json.loads(valid_json)

    if "cutSegments" not in data:
        print(f"[ERROR] 'cutSegments' missing in {llc_path}")
        return []
    return data["cutSegments"]

def slice_csv_for_segments(csv_path, segments, fps, clips_folder):
    """
    1) Loads the entire CSV (assumes frame_index is in the first column).
    2) For each segment in 'segments', convert (start,end) sec -> frames.
    3) Filter rows for frames in [start_frame, end_frame].
    4) Write them out to clip_1.csv, clip_2.csv, etc. in 'clips_folder'.
    """
    # Load entire CSV into memory
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Convert the first column (frame_index) to an integer
    for r in rows:
        r[0] = int(float(r[0]))  # e.g. "12.0" -> 12

    # For each segment, figure out the corresponding frame indices
    for i, seg in enumerate(segments, start=1):
        start_sec = seg["start"]
        end_sec   = seg["end"]

        start_frame = int(math.floor(start_sec * fps))
        end_frame   = int(math.floor(end_sec * fps))

        # Filter rows that are in [start_frame, end_frame]
        sliced_rows = [row for row in rows if start_frame <= row[0] <= end_frame]

        if not sliced_rows:
            print(f"  [WARN] No CSV rows found between frames {start_frame} - {end_frame}")
            continue

        # Write them to clip_#.csv in the same folder as clip_#.mp4
        csv_filename = f"clip_{i}.csv"
        out_path = os.path.join(clips_folder, csv_filename)

        with open(out_path, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)    # original CSV headers
            writer.writerows(sliced_rows)

        print(f"  [SAVED] {csv_filename} with {len(sliced_rows)} rows (frames {start_frame}-{end_frame})")

def process_video_folder(video_id):
    """
    - Looks for data/processed/video_<id>/video_<id>.llc
    - Looks for data/processed/video_<id>/video_<id>_normalized.csv
    - Slices CSV for each segment, storing clip_#.csv in
      data/processed/video_<id>/video_<id>_clips/ next to clip_#.mp4
    """
    folder_name = f"video_{video_id}"
    folder_path = os.path.join(PROCESSED_DIR, folder_name)

    llc_path = os.path.join(folder_path, f"{folder_name}.llc")
    csv_path = os.path.join(folder_path, f"{folder_name}_normalized.csv")

    # The folder for the existing MP4 clips
    clips_folder = os.path.join(folder_path, f"{folder_name}_clips")

    if not os.path.isfile(llc_path):
        print(f"[SKIP] No .llc in {folder_path}")
        return
    if not os.path.isfile(csv_path):
        print(f"[SKIP] No normalized CSV in {folder_path}")
        return
    if not os.path.isdir(clips_folder):
        print(f"[SKIP] No clips folder found: {clips_folder}")
        return

    segments = load_llc_segments(llc_path)
    if not segments:
        print(f"[INFO] .llc has no segments in {llc_path}")
        return

    print(f"[INFO] Splitting CSV for {folder_name} -> placing slices into {clips_folder}")
    slice_csv_for_segments(csv_path, segments, FPS, clips_folder)

def main():
    """
    Iterates over data/processed folders named like "video_x"
    and splits the associated normalized CSV for each segment in .llc.
    Output CSV slices are placed in the same folder as the MP4 clips:
      e.g., data/processed/video_3/video_3_clips/clip_1.csv
    """
    for name in os.listdir(PROCESSED_DIR):
        if not name.startswith("video_"):
            continue
        path = os.path.join(PROCESSED_DIR, name)
        if not os.path.isdir(path):
            continue

        # Extract the numeric ID, e.g. "3" from "video_3"
        try:
            vid_id = int(name.split("_")[1])
        except (IndexError, ValueError):
            print(f"[WARN] Unexpected folder name: {name}")
            continue

        process_video_folder(vid_id)

    print("[DONE] CSV slicing complete.")

if __name__ == "__main__":
    main()
