#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
postcutallinone.py

Combines four post-cut processing stages in one script:
  1) Feature Engineering - Adds stroke labels to frames based on LLC files
  2) Clip Generation     - Creates video clips from labeled segments
  3) Clip Collection     - Collects all clips into a single "Strokes_Library" folder
  4) Time Normalization  - Normalizes clip timelines to a fixed number of frames

Directory usage:
  videos/
      video_X/
          video_X.mp4               (original video)
          video_X_data.csv          (pose data from MediaPipe)
          video_X_normalized.csv    (normalized pose data)
          video_X.llc               (manual segmentation file)
          video_X_labeled.csv       (output from feature engineering)
          video_X_clips/            (output from clip generation)
              stroke_1.mp4
              stroke_2.mp4
              ...
  Strokes_Library/
      stroke_Y/
          stroke.csv               (output from clip collection)
          stroke_norm.csv          (output from time normalization)
          stroke_clip.mp4          (output from clip collection)
          stroke_overlay.mp4       (output from clip collection)
          stroke_skeleton.mp4      (output from clip collection)

Simply run:
    python postcutallinone.py
"""

import os
import csv
import json
import math
import re
import shutil
import ffmpeg
import numpy as np
import cv2  # Add import for OpenCV

# -------------------
# USER CONFIG
# -------------------
FPS = 30  # Frames per second
RESAMPLED_FRAMES = 120  # Number of frames after time normalization
FORCE_REPROCESS = False  # If True, re-process even if files exist

# Root directories (relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "..", "videos")  # New videos directory
FINAL_LIBRARY = os.path.join(SCRIPT_DIR, "..", "Strokes_Library")
OUTPUT_SUFFIX = "_norm.csv"  # Suffix for time-normalized files

# Add skeleton configuration constants near the top after other constants
# Skeleton config:
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

COLOR_LEFT = (255, 0, 0)    
COLOR_RIGHT = (0, 0, 255)   
COLOR_CENTER = (0, 255, 0)  
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

# Transparency in overlay mode (1.0 = no transparency)
ALPHA = 1.0

# -------------------------------------------------------------------------
# PART 1: FEATURE ENGINEERING - Add stroke labels based on LLC files
# -------------------------------------------------------------------------
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
    if os.path.exists(output_csv) and not FORCE_REPROCESS:
        print(f"[SKIP] Labeled CSV already exists: {output_csv}")
        return

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

def run_feature_engineering():
    """
    For each video folder in videos/, add stroke labels based on LLC files.
    """
    print("\n=== FEATURE ENGINEERING: Adding stroke labels ===")
    
    # Iterate over each subfolder in VIDEOS_DIR
    for folder in os.listdir(VIDEOS_DIR):
        folder_path = os.path.join(VIDEOS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Check if this is a video folder
        if not folder.startswith("video_"):
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
    
    print("=== FEATURE ENGINEERING COMPLETE ===\n")

# -------------------------------------------------------------------------
# PART 2: CLIP GENERATION - Create video clips from labeled segments
# -------------------------------------------------------------------------
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

        if os.path.exists(output_file) and not FORCE_REPROCESS:
            print(f"[SKIP] Clip already exists: {output_file}")
            continue

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

def extract_clip_csv(labeled_csv_path, segments, clips_folder):
    """
    For each segment, extract the corresponding rows from the labeled CSV
    and save them as a new CSV in the clips folder.
    """
    with open(labeled_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    for i, (start_frame, end_frame) in enumerate(segments):
        output_file = os.path.join(clips_folder, f"stroke_{i+1}.csv")
        
        if os.path.exists(output_file) and not FORCE_REPROCESS:
            print(f"[SKIP] Clip CSV already exists: {output_file}")
            continue
            
        clip_rows = []
        for row in rows:
            frame_idx = int(float(row[0]))
            if start_frame <= frame_idx <= end_frame:
                # Adjust frame index to start from 0 within the clip
                adjusted_row = [str(frame_idx - start_frame)] + row[1:]
                clip_rows.append(adjusted_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(clip_rows)
        
        print(f"[INFO] Created clip CSV: {output_file}")

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
    
    # Extract corresponding CSV segments
    extract_clip_csv(labeled_csv_path, segments, clips_folder_path)
    
    print(f"[DONE] All segments clipped into: {clips_folder_path}")

def run_clip_generation():
    """
    For each video folder in videos/, generate clips based on labeled segments.
    """
    print("\n=== CLIP GENERATION: Creating video clips ===")
    
    # Loop through all "video_x" folders under videos/
    for folder_name in os.listdir(VIDEOS_DIR):
        folder_path = os.path.join(VIDEOS_DIR, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith("video_"):
            continue

        # Attempt to process that folder if it matches our structure
        process_video_folder(folder_path)
    
    print("=== CLIP GENERATION COMPLETE ===\n")

# -------------------------------------------------------------------------
# PART 3: CLIP COLLECTION - Collect all clips into a single folder
# -------------------------------------------------------------------------
def run_clip_collection():
    """
    Collects all clip files (MP4, CSV) from videos/video_x/video_x_clips
    and places them into a structured "Strokes_Library" folder with subfolders for each stroke.
    """
    print("\n=== CLIP COLLECTION: Collecting all clips ===")
    
    # Create the Strokes Library folder if it doesn't exist
    os.makedirs(FINAL_LIBRARY, exist_ok=True)
    
    # Track the next stroke index
    stroke_counter = 1
    
    # Iterate over every subfolder in videos/
    for folder_name in os.listdir(VIDEOS_DIR):
        folder_path = os.path.join(VIDEOS_DIR, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith("video_"):
            continue

        # The clip folder is videos/video_x/video_x_clips
        clips_subfolder = os.path.join(folder_path, f"{folder_name}_clips")
        if not os.path.isdir(clips_subfolder):
            continue
            
        # Group files by stroke number
        stroke_files = {}
        for filename in os.listdir(clips_subfolder):
            if not os.path.isfile(os.path.join(clips_subfolder, filename)):
                continue
                
            # Extract stroke number from filename (e.g., "stroke_1.mp4" -> 1)
            match = re.match(r'stroke_(\d+)\.(\w+)', filename)
            if match:
                stroke_num = match.group(1)
                file_ext = match.group(2)
                
                if stroke_num not in stroke_files:
                    stroke_files[stroke_num] = []
                    
                stroke_files[stroke_num].append((filename, file_ext))
        
        # Process each stroke
        for stroke_num, files in stroke_files.items():
            # Create a stroke folder in the library
            stroke_folder = os.path.join(FINAL_LIBRARY, f"stroke_{stroke_counter}")
            os.makedirs(stroke_folder, exist_ok=True)
            
            # Create a reference file to track the source video
            source_info = os.path.join(stroke_folder, "source_info.txt")
            with open(source_info, 'w') as f:
                f.write(f"Source Video: {folder_name}\n")
                f.write(f"Stroke Number in Video: {stroke_num}\n")
            
            # Copy and rename files for this stroke
            for filename, file_ext in files:
                src_file = os.path.join(clips_subfolder, filename)
                
                # Determine the new filename based on extension
                if file_ext == "mp4":
                    dest_filename = "stroke_clip.mp4"
                elif file_ext == "csv":
                    dest_filename = "stroke.csv"
                else:
                    continue  # Skip unrecognized files
                    
                dest_path = os.path.join(stroke_folder, dest_filename)
                
                if os.path.exists(dest_path) and not FORCE_REPROCESS:
                    print(f"[SKIP] File already exists in Strokes Library: {dest_path}")
                    continue
                
                # Copy the file
                shutil.copy2(src_file, dest_path)
                print(f"[COPY] {src_file} -> {dest_path}")
                
                # Now we need to generate the overlay and skeleton videos
                if file_ext == "mp4":
                    # Path to original video
                    video_path = os.path.join(folder_path, f"{folder_name}.mp4")
                    
                    # Path to CSV file with pose data
                    csv_path = os.path.join(folder_path, f"{folder_name}_data.csv")
                    if not os.path.exists(csv_path):
                        print(f"[WARN] Pose data CSV not found: {csv_path}, using placeholders.")
                        # In case we don't have the CSV data, use placeholders like before
                        overlay_path = os.path.join(stroke_folder, "stroke_overlay.mp4")
                        skeleton_path = os.path.join(stroke_folder, "stroke_skeleton.mp4")
                        
                        if not os.path.exists(overlay_path) or FORCE_REPROCESS:
                            shutil.copy2(src_file, overlay_path)
                            print(f"[PLACEHOLDER] Created overlay video: {overlay_path}")
                        
                        if not os.path.exists(skeleton_path) or FORCE_REPROCESS:
                            shutil.copy2(src_file, skeleton_path)
                            print(f"[PLACEHOLDER] Created skeleton video: {skeleton_path}")
                    else:
                        # Generate real overlay and skeleton videos using CSV pose data
                        overlay_path = os.path.join(stroke_folder, "stroke_overlay.mp4")
                        skeleton_path = os.path.join(stroke_folder, "stroke_skeleton.mp4")
                        
                        # Create overlay video (skeleton over original video)
                        create_skeleton_video(src_file, csv_path, overlay_path, mode="overlay")
                        
                        # Create skeleton video (skeleton on black background)
                        create_skeleton_video(src_file, csv_path, skeleton_path, mode="skeleton")
            
            # Update the status file in the video folder to mark it as processed
            status_file = os.path.join(folder_path, "status.txt")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status_lines = f.readlines()
                
                updated_lines = []
                for line in status_lines:
                    if line.startswith("is_fully_processed:"):
                        updated_lines.append("is_fully_processed: True\n")
                    else:
                        updated_lines.append(line)
                
                with open(status_file, 'w') as f:
                    f.writelines(updated_lines)
            
            # Increment the stroke counter for the next stroke
            stroke_counter += 1
    
    print(f"[DONE] All clips consolidated into: {FINAL_LIBRARY}")
    print("=== CLIP COLLECTION COMPLETE ===\n")

# -------------------------------------------------------------------------
# PART 4: TIME NORMALIZATION - Normalize clip timelines
# -------------------------------------------------------------------------
def time_normalize_csv(csv_path, out_path, num_frames):
    """
    Resamples a CSV to have a fixed number of frames.
    """
    if os.path.exists(out_path) and not FORCE_REPROCESS:
        print(f"[SKIP] Normalized CSV already exists: {out_path}")
        return
        
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Convert all numeric columns to float
    # If the last column is non-numeric (e.g., swing_phase), we'll handle it separately
    last_col_is_numeric = True
    try:
        float(rows[0][-1])
    except (ValueError, IndexError):
        last_col_is_numeric = False

    numeric_header = header if last_col_is_numeric else header[:-1]
    numeric_data = []
    non_numeric_data = [] if not last_col_is_numeric else None
    
    for row in rows:
        if last_col_is_numeric:
            numeric_data.append([float(x) for x in row])
        else:
            numeric_data.append([float(x) for x in row[:-1]])
            non_numeric_data.append(row[-1])
    
    numeric_data = np.array(numeric_data)  # shape = (F, D)

    F = numeric_data.shape[0]
    if F == 0:
        print(f"[WARN] CSV is empty: {csv_path}")
        return

    # Create new indices from 0..F-1 in 'num_frames' steps
    old_max = F - 1
    new_indices = np.linspace(0, old_max, num_frames)

    D = numeric_data.shape[1]
    new_data = []
    # Interpolate each column individually
    for col_idx in range(D):
        col = numeric_data[:, col_idx]
        col_resampled = np.interp(new_indices, np.arange(F), col)
        new_data.append(col_resampled)
    # Transpose to shape (num_frames, D)
    new_data = np.array(new_data).T

    # Build output rows
    output_rows = []
    for i in range(num_frames):
        row_vals = [i] + list(new_data[i, 1:])  # Use i as the new frame index
        
        # If we had a non-numeric last column, we need to handle it
        # For simplicity, we'll use the value from the nearest original frame
        if not last_col_is_numeric:
            nearest_idx = int(round(new_indices[i]))
            nearest_idx = min(nearest_idx, len(non_numeric_data) - 1)
            row_vals.append(non_numeric_data[nearest_idx])
            
        output_rows.append(row_vals)

    # Adjust the header
    new_header = ["resampled_frame_idx"] + numeric_header[1:]
    if not last_col_is_numeric:
        new_header.append(header[-1])

    with open(out_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(new_header)
        writer.writerows(output_rows)

    print(f"[DONE] {csv_path} -> {out_path}")

def run_time_normalization():
    """
    For each stroke folder in the Strokes Library, normalize its CSV timeline.
    """
    print("\n=== TIME NORMALIZATION: Normalizing clip timelines ===")
    
    # Scan all stroke folders in the Strokes Library
    for folder_name in os.listdir(FINAL_LIBRARY):
        if not folder_name.startswith("stroke_"):
            continue
            
        folder_path = os.path.join(FINAL_LIBRARY, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Look for the stroke.csv file
        csv_path = os.path.join(folder_path, "stroke.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] CSV file not found in {folder_path}. Skipping.")
            continue
            
        # Output path
        out_path = os.path.join(folder_path, "stroke_norm.csv")
        
        # Time-normalize
        time_normalize_csv(csv_path, out_path, RESAMPLED_FRAMES)
    
    print("=== TIME NORMALIZATION COMPLETE ===\n")

# -------------------------------------------------------------------------
# MASTER "MAIN" - calls all steps
# -------------------------------------------------------------------------
def main():
    """
    Main function that runs all four processing steps in sequence.
    """
    print("=== POST-CUT PROCESSING PIPELINE ===")
    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Strokes Library: {FINAL_LIBRARY}")
    print(f"FPS: {FPS}")
    print(f"Resampled frames: {RESAMPLED_FRAMES}")
    print(f"Force reprocess: {FORCE_REPROCESS}")
    
    # Step 1: Feature Engineering - Add stroke labels
    run_feature_engineering()
    
    # Step 2: Clip Generation - Create video clips
    run_clip_generation()
    
    # Step 3: Clip Collection - Collect all clips
    run_clip_collection()
    
    # Step 4: Time Normalization - Normalize clip timelines
    run_time_normalization()
    
    print("\n=== ALL POST-CUT PROCESSING COMPLETE ===")
    print(f"Final clips and normalized CSVs are available in: {FINAL_LIBRARY}")
    
    # Show a summary of which videos have been processed
    print("\n=== VIDEO STATUS SUMMARY ===")
    processed_videos = 0
    ready_videos = 0
    for folder_name in os.listdir(VIDEOS_DIR):
        folder_path = os.path.join(VIDEOS_DIR, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith("video_"):
            continue
            
        status_file = os.path.join(folder_path, "status.txt")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_lines = f.read()
                
            if "is_fully_processed: True" in status_lines:
                print(f"✓ {folder_name} - Processed")
                processed_videos += 1
            elif "is_ready_for_clipping: True" in status_lines:
                print(f"! {folder_name} - Ready for processing but not yet processed")
                ready_videos += 1
            else:
                print(f"✗ {folder_name} - Missing required files")
    
    print(f"\nTotal videos: {len([d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d)) and d.startswith('video_')])}")
    print(f"Processed videos: {processed_videos}")
    print(f"Ready for processing: {ready_videos}")

# Add the following functions before run_clip_collection()
def load_landmarks(csv_path):
    """
    Reads CSV, returns a list of dict: frames[i][lm_id] = (x, y) in [0..1].
    Expects columns like 'lm_0_x', 'lm_0_y', etc.
    """
    frames = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Figure out which columns store each of the 33 landmarks
        lm_xy = {}
        for lm_id in range(33):
            x_col = f"lm_{lm_id}_x"
            y_col = f"lm_{lm_id}_y"
            if x_col in header and y_col in header:
                x_idx = header.index(x_col)
                y_idx = header.index(y_col)
                lm_xy[lm_id] = (x_idx, y_idx)

        for row in reader:
            landm = {}
            for lm_id in range(33):
                if lm_id in lm_xy:
                    x_c, y_c = lm_xy[lm_id]
                    x_val = float(row[x_c])
                    y_val = float(row[y_c])
                    landm[lm_id] = (x_val, y_val)
                else:
                    landm[lm_id] = (0.0, 0.0)
            frames.append(landm)
    return frames

def draw_pretty_skeleton(frame_bgr, landmarks_dict, width, height):
    """
    Draws a color-coded skeleton with thicker lines and anti-aliasing.
    """
    # Center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA * width), int(yA * height))
        ptB = (int(xB * width), int(yB * height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA * width), int(yA * height))
        ptB = (int(xB * width), int(yB * height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA * width), int(yA * height))
        ptB = (int(xB * width), int(yB * height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id in range(33):
        nx, ny = landmarks_dict[lm_id]
        px = int(nx * width)
        py = int(ny * height)
        # color-coded by side
        if lm_id in (11,13,15,23,25,27):
            kp_color = COLOR_LEFT
        elif lm_id in (12,14,16,24,26,28):
            kp_color = COLOR_RIGHT
        else:
            kp_color = COLOR_CENTER

        cv2.circle(frame_bgr, (px, py), CIRCLE_RADIUS, kp_color, -1, lineType=LINE_TYPE)

def create_skeleton_video(video_path, csv_path, output_path, mode="skeleton"):
    """
    Creates a skeleton or overlay video from a video and landmark CSV.
    - mode: "overlay" or "skeleton"
    """
    if os.path.exists(output_path) and not FORCE_REPROCESS:
        print(f"[SKIP] {mode} video already exists: {output_path}")
        return False
        
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}, skipping {mode} video")
        return False
        
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return False
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load landmarks
    frames_landmarks = load_landmarks(csv_path)
    print(f"[INFO] Found {len(frames_landmarks)} landmarks in {csv_path}")
    
    # Process frame by frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if mode == "skeleton":
            # Create black background
            output_frame = np.zeros((height, width, 3), dtype=np.uint8)
            if frame_idx < len(frames_landmarks):
                draw_pretty_skeleton(output_frame, frames_landmarks[frame_idx], width, height)
                
        elif mode == "overlay":
            output_frame = frame.copy()
            if frame_idx < len(frames_landmarks):
                if ALPHA < 1.0:
                    overlay = frame.copy()
                    draw_pretty_skeleton(overlay, frames_landmarks[frame_idx], width, height)
                    blend_factor = 1.0 - ALPHA
                    cv2.addWeighted(overlay, blend_factor, frame, ALPHA, 0, output_frame)
                else:
                    draw_pretty_skeleton(output_frame, frames_landmarks[frame_idx], width, height)
                    
        # Write frame
        out.write(output_frame)
        frame_idx += 1
        
    # Clean up
    cap.release()
    out.release()
    print(f"[INFO] Created {mode} video: {output_path}")
    return True

if __name__ == "__main__":
    main()
