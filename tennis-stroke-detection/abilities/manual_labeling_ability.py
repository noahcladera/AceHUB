#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
manual_labeling_ability.py

Handles importing, editing, or converting manually created stroke
labels (e.g., from an annotation interface). This version is aligned
with the LLC file structure:

{
  version: 1,
  mediaFileName: 'video_2.mp4',
  cutSegments: [
    {
      start: 0.474036,
      end: 3.560225,
      name: '',
    },
    ...
  ]
}
"""

import json
import os
import re

def load_llc_file(llc_path):
    """
    Loads manual cut segments from an LLC file and returns
    a Python dict with:
      {
        "version": 1,
        "mediaFileName": "video_2.mp4",
        "cutSegments": [
          {"start": 0.474036, "end": 3.560225, "name": ""},
          ...
        ]
      }

    :param llc_path: Path to the .llc file (semi-JSON format).
    :returns: A validated dict representing the LLC contents.
    :raises FileNotFoundError: if the .llc file does not exist.
    :raises ValueError: if the file cannot be parsed as JSON after conversion.
    """
    if not os.path.exists(llc_path):
        raise FileNotFoundError(f"[ERROR] LLC file not found: {llc_path}")

    with open(llc_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Convert to valid JSON if the file has single quotes or unquoted keys
    valid_text = convert_llc_to_valid_json(raw_text)
    try:
        data = json.loads(valid_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] Could not parse LLC file at {llc_path}: {e}")

    if "cutSegments" not in data:
        raise ValueError(f"[ERROR] 'cutSegments' key missing in LLC data: {llc_path}")

    return data

def convert_llc_to_valid_json(raw_text):
    """
    Converts LLC file text to valid JSON by:
      1) Wrapping unquoted keys with double quotes (e.g. version: => "version":).
      2) Replacing single quotes with double quotes.
      3) Removing trailing commas before } or ].

    :param raw_text: The raw contents of a .llc file as a string.
    :returns: A string with valid JSON formatting.
    """
    # 1) Wrap unquoted keys that appear before a colon
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', raw_text)

    # 2) Replace single quotes with double quotes
    #    This will make "name: ''" -> "name: \" \"" in JSON, for example
    text = text.replace("'", '"')

    # 3) Remove trailing commas before a closing brace/bracket
    text = re.sub(r',\s*(\}|])', r'\1', text)

    return text

def get_cut_segments(llc_data):
    """
    Retrieves the 'cutSegments' list from the LLC data, returning
    a list of dicts like:
       [
         {"start": 0.474036, "end": 3.560225, "name": ""},
         ...
       ]

    :param llc_data: The parsed LLC dict from load_llc_file.
    :returns: The list of segment dicts (each with 'start', 'end', 'name').
    """
    return llc_data.get("cutSegments", [])

def integrate_labels_with_csv(llc_data, csv_path, fps=30):
    """
    Example function to merge the cut segments from an LLC file
    (in seconds) with a CSV containing frame-level data. You might
    create a "stroke_label" column indicating which frames fall
    within any cut segment.

    :param llc_data: The parsed LLC dictionary from load_llc_file().
    :param csv_path: Path to the CSV file to be updated or validated.
    :param fps: Frames per second to convert seconds -> frame index.
    :returns: Optionally returns a structure or writes an updated CSV.
    """
    segments = get_cut_segments(llc_data)

    # Typically, you'd read the CSV, add a stroke_label column, and save it out.
    # This stub just prints debug info.
    print(f"[DEBUG] Merging LLC segments with CSV {csv_path} at {fps} FPS.")
    for seg in segments:
        start_frame = int(seg["start"] * fps)
        end_frame = int(seg["end"] * fps)
        print(f"  Segment: start={start_frame}, end={end_frame}, name='{seg['name']}'")

    # Actual merging logic depends on your existing CSV structure and
    # how you want to incorporate stroke labels.
    # See src/data/feature_engineering.py for a more complete example.
    pass