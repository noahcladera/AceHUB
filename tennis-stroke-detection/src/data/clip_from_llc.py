#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clip_from_llc.py

For each "video_x" folder in data/processed, we look for 'video_x.llc'.
We then read cutSegments (start/end times in seconds) and create short clips
from the raw MP4 in data/raw/video_x/video_x.mp4.

Output clips are saved in:
  data/processed/video_x/video_x_clips/
with filenames like "clip_1.mp4", "clip_2.mp4", etc.
"""

import os
import re
import json
import ffmpeg

# Adjust these paths if your directory structure is different
PROCESSED_DIR = os.path.join("data", "processed")
RAW_DIR = os.path.join("data", "raw")

def convert_llc_to_valid_json(raw_text):
    """
    Convert typical .llc text (with single quotes, unquoted keys, trailing commas)
    into valid JSON so we can parse it with `json.loads`.
    1) Wrap unquoted keys before a colon with double quotes.
    2) Replace single quotes with double quotes.
    3) Remove trailing commas before } or ].
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
    Reads the .llc file, converts to valid JSON, then returns a list of dicts:
      [
        { "start": float, "end": float, "name": "" },
        ...
      ]
    """
    with open(llc_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Convert to valid JSON if needed
    valid_text = convert_llc_to_valid_json(raw_text)
    data = json.loads(valid_text)

    if "cutSegments" not in data:
        print(f"[ERROR] 'cutSegments' field missing in {llc_path}")
        return []

    return data["cutSegments"]

def clip_segment(video_path, start_time, end_time, output_file):
    """
    Clips the segment [start_time, end_time] from video_path
    and saves to output_file using ffmpeg with copy codec (lossless).
    """
    duration = end_time - start_time
    if duration <= 0:
        print(f"  [SKIP] Invalid segment duration ({duration}).")
        return

    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .output(output_file, codec="copy", loglevel="error", y=None)
            .run()
        )
        print(f"  [CLIPPED] {output_file} ({start_time:.2f} - {end_time:.2f} sec)")
    except ffmpeg.Error as e:
        print(f"  [ERROR] ffmpeg failed on {output_file}: {e}")

def process_llc_for_video(video_id):
    """
    1) Looks for data/processed/video_<id>/video_<id>.llc
    2) Reads cut segments
    3) Finds raw MP4 in data/raw/video_<id>/video_<id>.mp4
    4) For each segment, clips to a new file under data/processed/video_<id>/video_<id>_clips
    """
    folder_name = f"video_{video_id}"
    llc_folder = os.path.join(PROCESSED_DIR, folder_name)
    llc_path   = os.path.join(llc_folder, f"{folder_name}.llc")

    raw_folder = os.path.join(RAW_DIR, folder_name)
    video_path = os.path.join(raw_folder, f"{folder_name}.mp4")

    if not os.path.isfile(llc_path):
        print(f"[SKIP] LLC not found: {llc_path}")
        return
    if not os.path.isfile(video_path):
        print(f"[SKIP] Raw video not found: {video_path}")
        return

    segments = load_llc_segments(llc_path)
    if not segments:
        print(f"[INFO] No segments found or empty .llc for {folder_name}.")
        return

    # Create an output folder for clips
    clips_folder = os.path.join(llc_folder, f"{folder_name}_clips")
    os.makedirs(clips_folder, exist_ok=True)

    print(f"[INFO] Found {len(segments)} segments in {llc_path}")
    # Clip each segment
    for i, seg in enumerate(segments, start=1):
        start_sec = seg["start"]
        end_sec   = seg["end"]
        clip_filename = f"clip_{i}.mp4"
        output_path   = os.path.join(clips_folder, clip_filename)
        clip_segment(video_path, start_sec, end_sec, output_path)

    print(f"[DONE] All clips saved to: {clips_folder}\n")

def main():
    """
    Iterates over each folder in data/processed named video_x,
    calls process_llc_for_video(x).
    """
    # E.g. check all subfolders in data/processed for "video_XX" pattern
    for name in os.listdir(PROCESSED_DIR):
        # Expect folders like "video_3", "video_10", etc.
        if not name.startswith("video_"):
            continue
        folder_path = os.path.join(PROCESSED_DIR, name)
        if not os.path.isdir(folder_path):
            continue

        # Extract the integer from "video_3" => 3
        # or you can just pass the full string without extracting
        # if your folder naming is guaranteed consistent.
        try:
            video_num = int(name.split("_")[1])
        except (IndexError, ValueError):
            print(f"[WARNING] Unexpected folder name: {name}, skipping.")
            continue

        process_llc_for_video(video_num)

if __name__ == "__main__":
    main()
