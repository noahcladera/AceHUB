#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
one_click_pipeline.py

A single "master" script that orchestrates:
  1) Asking for YouTube links.
  2) Downloading/Converting them (acquisition.py).
  3) Running pose_extraction.py for new videos (raw -> data.csv).
  4) Running normalization.py (data.csv -> normalized.csv).
  5) Pausing for the user to create .llc cuts in data/processed/video_X.
  6) Calling clip_from_llc.py to produce short MP4 segments + CSV slices.
  7) Optionally time-normalizing each clip CSV (time_normalize_clips.py).
  8) Optionally creating overlay/skeleton versions (auto_process_final_library.py).
  9) Finally collecting all into "Final library/" (collect_clips_flat.py).

This script uses "subprocess.run" to call each step. If you want more
fine-grained integration (like arguments or function calls), you'd refactor
the individual scripts to expose them as Python modules.

Usage:
  python one_click_pipeline.py
"""

import os
import subprocess
import sys
import time

def run_subprocess(cmd_list, check=True):
    """
    Utility to call a child script with nice printing.
    Example: run_subprocess(["python", "src/data/acquisition.py"]).
    """
    print(f"[INFO] Running: {' '.join(cmd_list)}")
    completed = subprocess.run(cmd_list, check=check)
    if completed.returncode != 0 and check:
        sys.exit(f"[ERROR] Command failed: {' '.join(cmd_list)}")
    return completed

def main():
    print("Welcome to the One-Click Pipeline!\n")

    # 1) Ask user for YouTube links (optional).
    print("Step 1: Download videos from YouTube.")
    # If you want to skip or if you have them already, press Enter with no links.
    links_input = input("Paste YouTube links (space or line separated), or leave blank to skip:\n")
    if links_input.strip():
        # Write them into acquisition.py or set them in environment:
        print("Updating acquisition script with your links (not implemented in detail).")
        # Typically you'd dynamically rewrite the youtube_urls in acquisition.py or pass them as an arg.
        # For simplicity, let's just instruct user to place them manually.
        print("Please open 'src/data/acquisition.py' and set 'youtube_urls' to your new links, then press Enter.")
        input("Press Enter once you're done updating acquisition.py with your new links.")
        
        # Now run acquisition
        run_subprocess(["python", "src/data/acquisition.py"])
    else:
        print("[SKIP] No new YouTube links entered. Move on.\n")

    # 2) Pose Extraction: run pose_extraction on data/raw -> data/interim
    print("\nStep 2: Pose Extraction (MediaPipe).")
    run_subprocess(["python", "src/data/pose_extraction.py"])

    # 3) Normalization: from data/interim -> data/processed
    print("\nStep 3: Normalization (rotating, scaling, smoothing).")
    run_subprocess(["python", "src/data/normalization.py"])

    # 4) Wait for user to produce .llc with manual cut segments
    print("\nStep 4: Manual .llc creation for each video_x folder in data/processed.")
    print("    a) For each data/processed/video_X, open the original raw MP4 in data/raw/video_X.")
    print("    b) Decide start/end times, write them in 'video_X.llc' file in data/processed/video_X.")
    input("Press Enter when you're done with .llc for all new videos, or skip if you have them.\n")

    # 5) Clip from LLC: produce short MP4 segments in data/processed/video_X_clips
    print("\nStep 5: Running clip_from_llc to do the actual cut -> short MP4 clips.")
    run_subprocess(["python", "src/data/clip_from_llc.py"])

    # 6) Ask if you want time normalization for each clip CSV
    ans_time = input("Do you want to time-normalize each clip CSV? (y/n): ").lower()
    if ans_time.startswith("y"):
        print("[INFO] We will flatten all existing clips first so they're in 'Final library', then time-normalize them.")
        # Possibly you might want to unify them after or you can do time-normalize directly in data/processed.
        # Let's do it AFTER we unify them in final library, so we get them all in one place:
        pass
    else:
        print("[SKIP] Time normalization.\n")

    # 7) Flatten all existing short clips from data/processed -> Final library
    print("\nStep 7: Flattening all short clips into the 'Final library' folder.")
    run_subprocess(["python", "src/data/collect_clips_flat.py"])

    # 8) If user wants to time-normalize each CSV in Final library
    if ans_time.startswith("y"):
        print("[INFO] Now performing time normalization on each CSV in 'Final library/'.")
        run_subprocess(["python", "src/data/time_normalize_clips.py"])

    # 9) Optionally create overlay / skeleton for each clip in 'Final library'
    ans_overlay = input("Do you want to create overlay & skeleton videos for each clip in 'Final library'? (y/n): ").lower()
    if ans_overlay.startswith("y"):
        print("[INFO] Running auto_process_final_library to create _overlay.mp4 & _skeleton.mp4 for each clip+csv.")
        run_subprocess(["python", "src/data/auto_process_final_library.py"])
    else:
        print("[SKIP] Overlay/skeleton creation.\n")

    # Done
    print("\nOne-click pipeline complete!")
    print("Check 'Final library/' for your consolidated short clips, optionally time-normalized, with CSVs.")
    print("You can also see any overlays or skeleton versions if you chose that step.")

if __name__ == "__main__":
    main()
