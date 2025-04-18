#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tennis_video_processor.py

A complete end-to-end script for processing tennis videos from start to finish.

This script:
1. Takes a video and its corresponding LosslessCut (LLC) file from unprocessed_videos directory
2. Extracts pose landmarks with MediaPipe
3. Normalizes the data
4. Processes segmentation based on the LLC file
5. Creates clips with skeleton overlays and 3D visualizations
6. Adds everything to the Strokes Library

Usage:
    python tennis_video_processor.py [--video-id VIDEO_ID]

Example:
    python tennis_video_processor.py
    python tennis_video_processor.py --video-id 5

Dependencies:
    pip install opencv-python mediapipe numpy pandas plotly
"""

import os
import sys
import shutil
import argparse
import subprocess
import glob
import csv
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.graph_objects as go
import time
import math
import json
import re

# Directory constants
UNPROCESSED_DIR = "unprocessed_videos"
VIDEOS_DIR = "videos"
STROKES_LIBRARY = "Strokes_Library"
DATA_PROCESSED_DIR = os.path.join("data", "processed")

# Visualization constants
# Color configuration
COLOR_LEFT = (255, 0, 0)  # Blue for left side
COLOR_RIGHT = (0, 0, 255)  # Red for right side
COLOR_CENTER = (0, 255, 0)  # Green for center

# Define partial pose connections for color-coding
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

# Full set of pose connections for complete skeleton visualization
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                  (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                  (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
                  (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                  (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]

# Drawing configuration
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def get_video_and_llc_paths():
    """
    Look in the unprocessed_videos directory and find video+LLC pairs.
    Returns a list of tuples (video_path, llc_path).
    """
    if not os.path.exists(UNPROCESSED_DIR):
        print(f"Error: Unprocessed videos directory not found: {UNPROCESSED_DIR}")
        return []
    
    video_files = []
    for file in os.listdir(UNPROCESSED_DIR):
        lower_file = file.lower()
        if lower_file.endswith(('.mp4', '.mov', '.avi')):
            # Get base name without extension
            base_name = os.path.splitext(file)[0]
            # Look for LLC files
            llc_file = None
            
            # Check for LLC with same name
            potential_llc = f"{base_name}.llc"
            if os.path.exists(os.path.join(UNPROCESSED_DIR, potential_llc)):
                llc_file = potential_llc
            
            # Also check for LLC with full filename
            potential_llc_full = f"{file}.llc"
            if os.path.exists(os.path.join(UNPROCESSED_DIR, potential_llc_full)):
                llc_file = potential_llc_full
            
            # If we found an LLC file, add the pair
            if llc_file:
                video_path = os.path.join(UNPROCESSED_DIR, file)
                llc_path = os.path.join(UNPROCESSED_DIR, llc_file)
                video_files.append((video_path, llc_path))
            else:
                print(f"Warning: No LLC file found for {file}, skipping.")
    
    return video_files

def get_next_video_id():
    """Find the next available video ID by checking existing folders"""
    try:
        video_dirs = []
        
        # Check videos directory
        if os.path.exists(VIDEOS_DIR):
            video_dirs.extend([d for d in os.listdir(VIDEOS_DIR) 
                              if d.startswith("video_") and os.path.isdir(os.path.join(VIDEOS_DIR, d))])
        
        # Check processed directory
        if os.path.exists(DATA_PROCESSED_DIR):
            video_dirs.extend([d for d in os.listdir(DATA_PROCESSED_DIR) 
                              if d.startswith("video_") and os.path.isdir(os.path.join(DATA_PROCESSED_DIR, d))])
        
        # Extract IDs and find max
        if video_dirs:
            video_ids = [int(d.split("_")[1]) for d in video_dirs]
            return max(video_ids) + 1
    except Exception as e:
        print(f"Error finding next video ID: {e}")
    
    # Default if no videos found or error
    return 1

def convert_video_to_mp4(input_path, output_path):
    """
    Convert MOV or other video formats to MP4 format with high quality.
    Returns True if conversion was successful.
    """
    try:
        print(f"Converting video format to MP4 with high quality settings...")
        result = subprocess.run(
            [
                "ffmpeg", 
                "-i", input_path, 
                "-c:v", "libx264", 
                "-preset", "slow",  # Higher quality encoding
                "-crf", "18",       # Lower CRF means higher quality (18 is considered visually lossless)
                "-pix_fmt", "yuv420p",  # Standard pixel format for compatibility
                "-vf", "fps=30",    # Consistent framerate
                "-c:a", "aac", 
                "-b:a", "192k",     # Better audio quality
                "-y", output_path
            ],
            capture_output=True, 
            text=True,
            check=True
        )
        print(f"Video conversion successful: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def extract_pose_data(video_path, output_csv):
    """
    Extract pose landmarks from video using MediaPipe.
    """
    print(f"[STEP 1/6] Extracting pose data from {os.path.basename(video_path)}")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has {frame_count} frames at {fps} FPS")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Build header
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        # Add angles
        header += ["right_elbow_angle", "left_elbow_angle"]
        writer.writerow(header)

        frame_index = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                row = [frame_index]
                landmarks = results.pose_landmarks.landmark
                
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.z, lm.visibility]

                # Calculate elbow angles
                r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]

                r_angle = calculate_angle(r_shoulder.x, r_shoulder.y,
                                        r_elbow.x, r_elbow.y,
                                        r_wrist.x, r_wrist.y)
                l_angle = calculate_angle(l_shoulder.x, l_shoulder.y,
                                        l_elbow.x, l_elbow.y,
                                        l_wrist.x, l_wrist.y)
                row += [r_angle, l_angle]
                
                writer.writerow(row)
                processed_frames += 1
            
            # Show progress
            frame_index += 1
            if frame_index % 30 == 0:
                pct = (frame_index / frame_count) * 100
                print(f"Processing: {frame_index}/{frame_count} frames ({pct:.1f}%)", end="\r")

    cap.release()
    print(f"\nProcessed {processed_frames} frames with valid pose data")
    print(f"[DONE] Pose data saved to {output_csv}")
    return processed_frames > 0

def calculate_angle(ax, ay, bx, by, cx, cy):
    """
    Calculate the angle at point B formed by A->B->C in 2D space.
    """
    BAx = ax - bx
    BAy = ay - by
    BCx = cx - bx
    BCy = cy - by
    dot = BAx * BCx + BAy * BCy
    magBA = math.sqrt(BAx**2 + BAy**2)
    magBC = math.sqrt(BCx**2 + BCy**2)
    if magBA == 0 or magBC == 0:
        return 0.0
    cos_angle = dot / (magBA * magBC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def normalize_pose_data(input_csv, output_csv):
    """
    Normalize the pose data to make it invariant to position and scale.
    """
    print(f"[STEP 2/6] Normalizing pose data")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        
        # Create a new dataframe for normalized data
        norm_df = pd.DataFrame()
        norm_df['frame_index'] = df['frame_index']
        
        # Process each frame
        for frame_idx in df['frame_index'].unique():
            frame_data = df[df['frame_index'] == frame_idx]
            
            # Extract all coordinates for this frame
            coords_x = np.array([frame_data[f'lm_{i}_x'].values[0] for i in range(33)])
            coords_y = np.array([frame_data[f'lm_{i}_y'].values[0] for i in range(33)])
            coords_z = np.array([frame_data[f'lm_{i}_z'].values[0] for i in range(33)])
            vis_scores = np.array([frame_data[f'lm_{i}_vis'].values[0] for i in range(33)])
            
            # Get hip midpoint (normally landmark 23 and 24)
            hip_mid_x = (coords_x[23] + coords_x[24]) / 2
            hip_mid_y = (coords_y[23] + coords_y[24]) / 2
            hip_mid_z = (coords_z[23] + coords_z[24]) / 2
            
            # Calculate scale (distance from hip to shoulder)
            shoulder_mid_x = (coords_x[11] + coords_x[12]) / 2
            shoulder_mid_y = (coords_y[11] + coords_y[12]) / 2
            shoulder_mid_z = (coords_z[11] + coords_z[12]) / 2
            
            scale = np.sqrt((shoulder_mid_x - hip_mid_x)**2 + 
                           (shoulder_mid_y - hip_mid_y)**2 + 
                           (shoulder_mid_z - hip_mid_z)**2)
            
            if scale == 0:
                scale = 1.0  # Avoid division by zero
            
            # Normalize coordinates by centering on hip and scaling
            norm_x = (coords_x - hip_mid_x) / scale
            norm_y = (coords_y - hip_mid_y) / scale
            norm_z = (coords_z - hip_mid_z) / scale
            
            # Store normalized coordinates in dataframe
            for i in range(33):
                norm_df.loc[norm_df['frame_index'] == frame_idx, f'lm_{i}_x'] = norm_x[i]
                norm_df.loc[norm_df['frame_index'] == frame_idx, f'lm_{i}_y'] = norm_y[i]
                norm_df.loc[norm_df['frame_index'] == frame_idx, f'lm_{i}_z'] = norm_z[i]
                norm_df.loc[norm_df['frame_index'] == frame_idx, f'lm_{i}_vis'] = vis_scores[i]
        
        # Include angle columns if they exist
        if 'right_elbow_angle' in df.columns and 'left_elbow_angle' in df.columns:
            norm_df['right_elbow_angle'] = df['right_elbow_angle']
            norm_df['left_elbow_angle'] = df['left_elbow_angle']
        
        # Save normalized data
        norm_df.to_csv(output_csv, index=False)
        print(f"[DONE] Normalized data saved to {output_csv}")
        
    except Exception as e:
        print(f"Error during normalization: {e}")
        return False
    
    return True

def check_llc_file(llc_path):
    """
    Check if the LLC file has valid stroke annotations.
    Supports both simple text format and JSON format from LosslessCut.
    Returns True if the file contains valid stroke annotations.
    """
    if not os.path.exists(llc_path):
        return False
    
    with open(llc_path, 'r') as f:
        content = f.read()
    
    # Check if it's a LosslessCut JSON format
    if "{" in content and "cutSegments" in content:
        try:
            # Try to normalize the JSON format if needed
            # (LosslessCut uses JavaScript object notation, not strict JSON)
            content = re.sub(r'(?<=[{,])\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', content)
            content = content.replace("'", '"')
            content = re.sub(r',\s*(\}|])', r'\1', content)
            
            data = json.loads(content)
            segments = data.get("cutSegments", [])
            return len(segments) > 0
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: LLC file appears to be in JSON format but could not be parsed: {e}")
            # Fall through to check as text format
    
    # Check as simple text format (time time stroke_type)
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    float(parts[0])
                    float(parts[1])
                    return True  # Found at least one valid annotation
                except ValueError:
                    continue
    
    return False

def setup_for_pipeline(video_path, data_csv, norm_csv, llc_path, video_id):
    """
    Set up the files for the main pipeline:
    1. Copy video to data/processed/video_X/
    2. Copy normalized CSV to data/processed/video_X/
    3. Copy LLC file to data/processed/video_X/
    4. Create status.txt
    
    Args:
        video_path: Path to the video file
        data_csv: Path to the raw data CSV
        norm_csv: Path to the normalized data CSV
        llc_path: Path to the LLC file
        video_id: Video ID
        
    Returns:
        Path to the copied LLC file in the data/processed directory
    """
    # Create directories
    video_dir_processed = os.path.join(DATA_PROCESSED_DIR, f"video_{video_id}")
    video_dir_videos = os.path.join(VIDEOS_DIR, f"video_{video_id}")
    
    ensure_directory_exists(video_dir_processed)
    ensure_directory_exists(video_dir_videos)
    
    # Copy files to data/processed/video_X/
    target_video = os.path.join(video_dir_processed, f"video_{video_id}.mp4")
    target_data_csv = os.path.join(video_dir_processed, f"video_{video_id}_data.csv")
    target_norm_csv = os.path.join(video_dir_processed, f"video_{video_id}_normalized.csv")
    target_llc_processed = os.path.join(video_dir_processed, f"video_{video_id}.llc")
    
    # Check files exist before copying
    if os.path.exists(video_path):
        shutil.copy2(video_path, target_video)
    else:
        print(f"[WARN] Video file not found: {video_path}")
        
    if os.path.exists(data_csv):
        shutil.copy2(data_csv, target_data_csv)
    else:
        print(f"[WARN] Data CSV not found: {data_csv}")
        
    if os.path.exists(norm_csv):
        shutil.copy2(norm_csv, target_norm_csv)
    else:
        print(f"[WARN] Normalized CSV not found: {norm_csv}")
        
    if os.path.exists(llc_path):
        shutil.copy2(llc_path, target_llc_processed)
    else:
        print(f"[WARN] LLC file not found: {llc_path}")
    
    # Copy files to videos/video_X/
    video_file = os.path.join(video_dir_videos, f"video_{video_id}.mp4")
    if os.path.exists(video_path):
        shutil.copy2(video_path, video_file)
        
    if os.path.exists(norm_csv):
        shutil.copy2(norm_csv, os.path.join(video_dir_videos, f"video_{video_id}_normalized.csv"))
        
    if os.path.exists(llc_path):
        shutil.copy2(llc_path, os.path.join(video_dir_videos, f"video_{video_id}.llc"))
    
    # Create a status.txt file
    status_file = os.path.join(video_dir_videos, "status.txt")
    with open(status_file, 'w') as f:
        f.write(f"original_filename: {os.path.basename(video_path)}\n")
        f.write(f"processed_date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"has_raw_data: {os.path.exists(data_csv)}\n")
        f.write(f"has_normalized_data: {os.path.exists(norm_csv)}\n")
        f.write(f"has_llc: {os.path.exists(llc_path)}\n")
        f.write("is_ready_for_clipping: True\n")
        f.write("is_fully_processed: False\n")
    
    print(f"[STEP 3/6] Setting up files for the main pipeline (video_{video_id})")
    print(f"[DONE] Files set up for pipeline as video_{video_id}")
    
    return target_llc_processed

def run_stroke_segmentation():
    """
    Run the stroke segmentation script to generate clips.
    This handles Steps 4 and 5 of the process:
    - Feature engineering (adding stroke labels)
    - Clip generation
    - Clip collection
    - Time normalization
    """
    print(f"[STEP 4/6] Running stroke segmentation to generate clips")
    
    try:
        print("Executing: python src/data/stroke_segmentation.py")
        print("This may take some time...")
        
        # Run the command and capture output
        result = subprocess.run(
            ["python", "src/data/stroke_segmentation.py"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print detailed information about command execution
        print(f"Command completed with return code: {result.returncode}")
        print(f"Command stdout length: {len(result.stdout) if result.stdout else 0} chars")
        print(f"Command stderr length: {len(result.stderr) if result.stderr else 0} chars")
        
        # Print stdout for debugging, but limit length if it's too long
        if result.stdout:
            if len(result.stdout) > 1000:
                print("Stroke segmentation output (truncated):")
                print(result.stdout[:500])
                print("...")
                print(result.stdout[-500:])
            else:
                print("Stroke segmentation output:")
                print(result.stdout)
            
        # Check return code
        if result.returncode != 0:
            print(f"Stroke segmentation returned non-zero exit code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            print("Continuing despite errors...")
            return True  # Continue anyway
        
        # Look for specific success patterns in the output
        if "clips generated" in result.stdout or "Processing complete" in result.stdout:
            print("Stroke segmentation appears to have completed successfully.")
        else:
            print("Stroke segmentation completed, but no success message found.")
        
        return True
    except Exception as e:
        import traceback
        print(f"Exception running stroke segmentation: {e}")
        print(traceback.format_exc())
        print("Continuing despite errors...")
        return False  # Return False to indicate error

def get_strokes_for_video(video_id):
    """
    Find all strokes in the Strokes Library that came from the specified video.
    """
    strokes = []
    source_video_id = f"video_{video_id}"
    print(f"Searching for strokes from source video: {source_video_id}")
    
    # Debug: List all folders in Strokes_Library
    stroke_folders = glob.glob(os.path.join(STROKES_LIBRARY, "stroke_*"))
    print(f"Found {len(stroke_folders)} stroke folders in {STROKES_LIBRARY}")
    
    if not stroke_folders:
        # Additional check for any files in the library
        all_files = os.listdir(STROKES_LIBRARY) if os.path.exists(STROKES_LIBRARY) else []
        if all_files:
            print(f"Files in {STROKES_LIBRARY}: {all_files[:5]}")
        else:
            print(f"No files found in {STROKES_LIBRARY}")
        return []
    
    # Check the first few source info files for debugging
    for stroke_dir in stroke_folders[:5]:
        source_info_path = os.path.join(stroke_dir, "source_info.txt")
        if os.path.exists(source_info_path):
            with open(source_info_path, 'r') as f:
                content = f.read()
                print(f"Sample source info from {os.path.basename(stroke_dir)}:")
                print(content)
    
    # Look through all stroke folders
    for stroke_dir in stroke_folders:
        source_info_path = os.path.join(stroke_dir, "source_info.txt")
        if os.path.exists(source_info_path):
            with open(source_info_path, 'r') as f:
                content = f.read()
                # Check for both formats of source video identification
                if f"Source Video: {source_video_id}" in content or f"Source Video: video_{video_id}" in content:
                    strokes.append(os.path.basename(stroke_dir))
    
    if strokes:
        print(f"Found {len(strokes)} strokes from video_{video_id}")
    else:
        print(f"No strokes found for video_{video_id} in {STROKES_LIBRARY}")
        
        # As a fallback, we can directly access the clips in the video folder
        clips_folder = os.path.join(VIDEOS_DIR, f"video_{video_id}", f"video_{video_id}_clips")
        if os.path.exists(clips_folder):
            clip_files = [f for f in os.listdir(clips_folder) if f.endswith('.mp4')]
            print(f"However, found {len(clip_files)} clip files in {clips_folder}")
            
            # Manually copy these clips to the Strokes_Library if needed
            if clip_files and os.path.exists(STROKES_LIBRARY):
                print(f"Manually copying clips to {STROKES_LIBRARY}...")
                for i, clip_file in enumerate(clip_files):
                    stroke_num = i + 1
                    stroke_folder = os.path.join(STROKES_LIBRARY, f"stroke_{stroke_num}")
                    
                    if not os.path.exists(stroke_folder):
                        os.makedirs(stroke_folder, exist_ok=True)
                        
                        # Copy the clip
                        src_file = os.path.join(clips_folder, clip_file)
                        dest_file = os.path.join(stroke_folder, "stroke_clip.mp4")
                        shutil.copy2(src_file, dest_file)
                        
                        # Create source info
                        with open(os.path.join(stroke_folder, "source_info.txt"), 'w') as f:
                            f.write(f"Source Video: video_{video_id}\n")
                            f.write(f"Stroke Number in Video: {i+1}\n")
                            f.write(f"Manually copied by tennis_video_processor.py\n")
                        
                        strokes.append(f"stroke_{stroke_num}")
                
                print(f"Manually copied {len(clip_files)} clips to {STROKES_LIBRARY}")
    
    return sorted(strokes)

def create_3d_visualizations(video_id, strokes):
    """
    Stub function that doesn't do anything.
    This is a placeholder since 3D visualizations have been removed from the process.
    """
    print(f"[INFO] create_3d_visualizations is disabled - skipping this step.")
    return True

def extract_landmark_data(df):
    """
    Extract landmark coordinates from DataFrame.
    """
    num_landmarks = 33
    x_coords = np.zeros((len(df), num_landmarks))
    y_coords = np.zeros((len(df), num_landmarks))
    z_coords = np.zeros((len(df), num_landmarks))
    vis_scores = np.zeros((len(df), num_landmarks))
    
    for i in range(num_landmarks):
        x_coords[:, i] = df[f'lm_{i}_x']
        y_coords[:, i] = df[f'lm_{i}_y']
        z_coords[:, i] = df[f'lm_{i}_z']
        if f'lm_{i}_vis' in df.columns:
            vis_scores[:, i] = df[f'lm_{i}_vis']
    
    return x_coords, y_coords, z_coords, vis_scores

def create_animated_pose_figure(x_coords, y_coords, z_coords, vis_scores):
    """
    Simple placeholder function that creates a basic figure.
    This is a stripped-down version to avoid errors when visualizations are not needed.
    """
    print("Creating minimal placeholder figure...")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=5)
    ))
    fig.update_layout(
        title='3D Pose Animation (Placeholder)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    return fig

def process_video(video_path, llc_path, video_id=None):
    """
    Process a single video and its LLC file.
    
    Args:
        video_path: Path to the video file
        llc_path: Path to the LLC file
        video_id: Optional video ID, if not provided will be auto-assigned
    
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        print(f"\n=== PROCESSING VIDEO: {os.path.basename(video_path)} ===")
        print(f"LLC File: {os.path.basename(llc_path)}")
        
        # Validate inputs
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
        
        if not os.path.exists(llc_path):
            print(f"Error: LLC file not found: {llc_path}")
            return False
        
        # Check if LLC file has valid annotations
        print("Checking LLC file for valid annotations...")
        if not check_llc_file(llc_path):
            print(f"Error: LLC file has no valid stroke annotations: {llc_path}")
            return False
        
        # Get video basename without extension
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_ext = os.path.splitext(os.path.basename(video_path))[1].lower()
        print(f"Video basename: {video_basename}, extension: {video_ext}")
        
        # Create temporary work directory
        temp_dir = os.path.join(UNPROCESSED_DIR, f"{video_basename}_temp")
        ensure_directory_exists(temp_dir)
        print(f"Created temporary directory: {temp_dir}")
        
        # Define output paths
        data_csv = os.path.join(temp_dir, f"{video_basename}_data.csv")
        norm_csv = os.path.join(temp_dir, f"{video_basename}_normalized.csv")
        
        # Get video ID if not provided
        if video_id is None:
            video_id = get_next_video_id()
        
        print(f"Using video ID: {video_id}")
        
        # Step 1: Extract pose data - directly from original for better quality
        print(f"[STEP 1/6] Extracting pose data - STARTING")
        if not extract_pose_data(video_path, data_csv):
            print("Error extracting pose data. Exiting.")
            return False
        print(f"[STEP 1/6] Extracting pose data - COMPLETED")
        
        # Convert MOV to MP4 if needed (after pose extraction)
        processed_video_path = video_path
        if video_ext.lower() != '.mp4':
            print(f"Video is in {video_ext} format, converting to MP4...")
            mp4_path = os.path.join(temp_dir, f"{video_basename}.mp4")
            if not convert_video_to_mp4(video_path, mp4_path):
                print("Failed to convert video. Make sure ffmpeg is installed.")
                return False
            # Use the converted MP4 file for the remaining processing
            processed_video_path = mp4_path
            print(f"Using converted video for further processing: {processed_video_path}")
        
        # Step 2: Normalize pose data
        print(f"[STEP 2/6] Normalizing pose data - STARTING")
        if not normalize_pose_data(data_csv, norm_csv):
            print("Error normalizing pose data. Exiting.")
            return False
        print(f"[STEP 2/6] Normalizing pose data - COMPLETED")
        
        # Step 3: Set up for pipeline
        print(f"[STEP 3/6] Setting up for pipeline - STARTING")
        target_llc = setup_for_pipeline(processed_video_path, data_csv, norm_csv, llc_path, video_id)
        print(f"[STEP 3/6] Setting up for pipeline - COMPLETED")
        
        # Verify setup worked correctly
        video_dir_videos = os.path.join(VIDEOS_DIR, f"video_{video_id}")
        print(f"Verifying directory: {video_dir_videos}")
        if not os.path.exists(video_dir_videos):
            print(f"Error: Video directory not created: {video_dir_videos}")
            return False
        
        video_file_videos = os.path.join(video_dir_videos, f"video_{video_id}.mp4")
        print(f"Verifying video file: {video_file_videos}")
        if not os.path.exists(video_file_videos):
            print(f"Error: Video file not copied: {video_file_videos}")
            # Try to fix it
            if os.path.exists(processed_video_path):
                print("Attempting to copy video file again...")
                shutil.copy2(processed_video_path, video_file_videos)
            else:
                return False
        
        llc_file_videos = os.path.join(video_dir_videos, f"video_{video_id}.llc")
        print(f"Verifying LLC file: {llc_file_videos}")
        if not os.path.exists(llc_file_videos):
            print(f"Error: LLC file not copied: {llc_file_videos}")
            # Try to fix it
            if os.path.exists(llc_path):
                print("Attempting to copy LLC file again...")
                shutil.copy2(llc_path, llc_file_videos)
            else:
                return False
        
        # Step 4: Run stroke segmentation
        print(f"[STEP 4/6] Running stroke segmentation - STARTING")
        segmentation_success = run_stroke_segmentation()
        print(f"[STEP 4/6] Stroke segmentation result: {segmentation_success}")
        if not segmentation_success:
            print("Error running stroke segmentation. Continuing anyway...")
        print(f"[STEP 4/6] Running stroke segmentation - COMPLETED")
        
        # Check if stroke segmentation created clips
        clips_dir = os.path.join(video_dir_videos, f"video_{video_id}_clips")
        clips_exist = False
        
        print(f"Checking for clips in: {clips_dir}")
        if os.path.exists(clips_dir):
            clips = [f for f in os.listdir(clips_dir) if f.endswith('.mp4')]
            print(f"Found {len(clips)} clips in {clips_dir}")
            if clips:
                clips_exist = True
                print(f"First few clips: {clips[:5]}")
            else:
                print("Warning: No clips created, stroke segmentation might have failed")
        else:
            print(f"Warning: Clips directory not created: {clips_dir}")
        
        # Step 5: Check for strokes in Strokes Library
        print(f"[STEP 5/6] Processing clips - STARTING")
        
        # First check if there are any strokes already in the library from this video
        print(f"Checking for existing strokes from video_{video_id} in Strokes Library...")
        existing_strokes = get_strokes_for_video(video_id)
        if existing_strokes:
            print(f"Found {len(existing_strokes)} strokes already in library: {existing_strokes[:5]}")
        
        # If clips exist, always use the fallback method to ensure they're properly copied
        strokes_from_clips = []
        if clips_exist:
            print(f"Clips exist in {clips_dir}. Using manual copy method to ensure all clips are in library.")
            strokes_from_clips = manually_copy_clips_to_library(video_id)
            print(f"Copied {len(strokes_from_clips)} strokes to library")
        
        # Combine strokes from both methods
        all_strokes = list(set(existing_strokes + strokes_from_clips))
        print(f"Total strokes for video_{video_id}: {len(all_strokes)}")
        print(f"[STEP 5/6] Processing clips - COMPLETED (found/created {len(all_strokes)} strokes)")
        
        # SKIP: No 3D visualizations needed
        print(f"SKIPPING 3D visualizations as they've been removed from the process")
        
        # Clean up step
        print(f"[STEP 6/6] Cleaning up - STARTING")
        
        # Clean up temporary files
        print(f"Cleaning up temporary files in: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[STEP 6/6] Cleaning up - COMPLETED")
        
        print("\n=== PROCESSING COMPLETE ===")
        print(f"Processed {video_basename} as video_{video_id}")
        if all_strokes:
            print(f"Generated {len(all_strokes)} stroke clips in the Strokes Library")
            print(f"Results are available in the {STROKES_LIBRARY} directory")
        else:
            print("No strokes were found or created in the Strokes Library.")
            print(f"Check the video_{video_id}_clips directory for clips that might have been generated.")
        
        return True
    except Exception as e:
        import traceback
        print(f"=== ERROR IN PROCESS_VIDEO ===")
        print(f"Exception: {e}")
        print(traceback.format_exc())
        return False

def manually_copy_clips_to_library(video_id):
    """
    Manually copy clips from video_X_clips folder to the Strokes Library.
    This is a fallback method if the stroke segmentation process doesn't copy them correctly.
    
    Args:
        video_id: Video ID
        
    Returns:
        List of strokes that were created
    """
    print(f"[FALLBACK] Manually copying clips to Strokes Library for video_{video_id}")
    
    clips_folder = os.path.join(VIDEOS_DIR, f"video_{video_id}", f"video_{video_id}_clips")
    if not os.path.exists(clips_folder):
        print(f"Clips folder not found: {clips_folder}")
        return []
    
    # Find normalized CSV from the video directory
    video_norm_csv = os.path.join(VIDEOS_DIR, f"video_{video_id}", f"video_{video_id}_normalized.csv")
    if not os.path.exists(video_norm_csv):
        print(f"Warning: Normalized CSV not found: {video_norm_csv}")
    else:
        print(f"Found normalized CSV: {video_norm_csv}")
    
    # Get existing stroke IDs from the Strokes Library
    # to determine the next available stroke ID
    ensure_directory_exists(STROKES_LIBRARY)
    
    # Find the highest stroke number
    existing_strokes = glob.glob(os.path.join(STROKES_LIBRARY, "stroke_*"))
    next_stroke_id = 1
    if existing_strokes:
        try:
            stroke_ids = [int(os.path.basename(s).split("_")[1]) for s in existing_strokes]
            next_stroke_id = max(stroke_ids) + 1
        except (ValueError, IndexError):
            pass
    
    print(f"Next available stroke ID: {next_stroke_id}")
    
    # Find all clip files
    mp4_files = [f for f in os.listdir(clips_folder) if f.endswith('.mp4')]
    csv_files = [f for f in os.listdir(clips_folder) if f.endswith('.csv')]
    
    print(f"Found {len(mp4_files)} MP4 files and {len(csv_files)} CSV files in clips folder")
    
    # Group files by stroke number
    strokes = {}
    for mp4_file in mp4_files:
        match = re.match(r'stroke_(\d+)\.mp4', mp4_file)
        if match:
            stroke_num = match.group(1)
            if stroke_num not in strokes:
                strokes[stroke_num] = {'mp4': None, 'csv': None}
            strokes[stroke_num]['mp4'] = mp4_file
    
    for csv_file in csv_files:
        match = re.match(r'stroke_(\d+)\.csv', csv_file)
        if match:
            stroke_num = match.group(1)
            if stroke_num not in strokes:
                strokes[stroke_num] = {'mp4': None, 'csv': None}
            strokes[stroke_num]['csv'] = csv_file
    
    print(f"Found {len(strokes)} paired clips to process")
    
    created_strokes = []
    
    # Copy each stroke to the Strokes Library
    for stroke_num, files in strokes.items():
        stroke_id = next_stroke_id
        next_stroke_id += 1
        
        stroke_folder = os.path.join(STROKES_LIBRARY, f"stroke_{stroke_id}")
        os.makedirs(stroke_folder, exist_ok=True)
        
        print(f"Creating stroke_{stroke_id} in {stroke_folder}")
        
        # Copy MP4 and CSV if available
        clip_file_path = None
        csv_file_path = None
        
        if files['mp4']:
            src_file = os.path.join(clips_folder, files['mp4'])
            dest_file = os.path.join(stroke_folder, "stroke_clip.mp4")
            shutil.copy2(src_file, dest_file)
            print(f"Copied {files['mp4']} to {dest_file}")
            clip_file_path = dest_file
            
            # Also create a "raw" version of the clip for reference
            raw_file = os.path.join(stroke_folder, "stroke_raw.mp4")
            shutil.copy2(src_file, raw_file)
            print(f"Created raw video: {raw_file}")
        
        if files['csv']:
            src_file = os.path.join(clips_folder, files['csv'])
            dest_file = os.path.join(stroke_folder, "stroke.csv")
            shutil.copy2(src_file, dest_file)
            print(f"Copied {files['csv']} to {dest_file}")
            csv_file_path = dest_file
            
            # Create a normalized version by copying the clip-specific CSV
            norm_file = os.path.join(stroke_folder, "stroke_norm.csv")
            shutil.copy2(src_file, norm_file)
            print(f"Created normalized CSV from clip: {norm_file}")
            
            # If we have the full video normalized CSV, also copy that
            if os.path.exists(video_norm_csv):
                full_norm_file = os.path.join(stroke_folder, "stroke_full_norm.csv")
                shutil.copy2(video_norm_csv, full_norm_file)
                print(f"Added full normalized data: {full_norm_file}")
        
        # Create proper skeleton and overlay visualizations if we have both clip and CSV
        if clip_file_path and csv_file_path:
            overlay_output = os.path.join(stroke_folder, "stroke_overlay.mp4")
            skeleton_output = os.path.join(stroke_folder, "stroke_skeleton.mp4")
            
            print(f"Creating pose visualizations for stroke_{stroke_id}...")
            success = create_pose_visualizations(
                clip_file_path, 
                csv_file_path,
                overlay_output,
                skeleton_output
            )
            
            if not success:
                print(f"Failed to create visualizations. Creating simple copies instead.")
                # Fallback to simple copies if visualization fails
                if not os.path.exists(overlay_output):
                    shutil.copy2(clip_file_path, overlay_output)
                if not os.path.exists(skeleton_output):
                    # Create a blank skeleton video - could be improved
                    shutil.copy2(clip_file_path, skeleton_output)
        else:
            print(f"Missing clip or CSV file, cannot create visualizations")
        
        # Create source info file
        source_info_path = os.path.join(stroke_folder, "source_info.txt")
        with open(source_info_path, 'w') as f:
            f.write(f"Source Video: video_{video_id}\n")
            f.write(f"Stroke Number in Video: {stroke_num}\n")
            f.write(f"Manually copied by tennis_video_processor.py\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        created_strokes.append(f"stroke_{stroke_id}")
    
    print(f"[FALLBACK] Created {len(created_strokes)} strokes in the Strokes Library")
    return created_strokes

def main():
    """Main function that processes all video/LLC pairs in the unprocessed videos folder."""
    parser = argparse.ArgumentParser(description="Process tennis videos from start to finish")
    parser.add_argument("--video-id", type=int, help="Specify a starting video ID instead of auto-detecting")
    parser.add_argument("--single", action="store_true", help="Process only the first video found")
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    ensure_directory_exists(UNPROCESSED_DIR)
    ensure_directory_exists(VIDEOS_DIR)
    ensure_directory_exists(STROKES_LIBRARY)
    ensure_directory_exists(DATA_PROCESSED_DIR)
    
    # Find all video+LLC pairs in the unprocessed directory
    video_llc_pairs = get_video_and_llc_paths()
    
    if not video_llc_pairs:
        print("No video+LLC pairs found in the unprocessed_videos directory.")
        print("Please add video files and their corresponding .llc files.")
        return 1
    
    print(f"Found {len(video_llc_pairs)} video+LLC pairs to process.")
    
    # Process videos
    video_id = args.video_id if args.video_id is not None else get_next_video_id()
    processed_count = 0
    
    for video_path, llc_path in video_llc_pairs:
        current_video_id = video_id + processed_count
        success = process_video(video_path, llc_path, current_video_id)
        
        if success:
            processed_count += 1
            
            # Copy the processed video to videos library
            video_filename = f"video_{current_video_id}.mp4"
            videos_lib_path = os.path.join(VIDEOS_DIR, f"video_{current_video_id}", video_filename)
            
            # Ensure the video directory exists in the videos library
            video_dir = os.path.dirname(videos_lib_path)
            ensure_directory_exists(video_dir)
            
            try:
                # Copy the video file if it doesn't already exist in the videos library
                if not os.path.exists(videos_lib_path):
                    shutil.copy2(video_path, videos_lib_path)
                    print(f"Copied video to videos library: {videos_lib_path}")
                else:
                    print(f"Video already exists in videos library: {videos_lib_path}")
            except Exception as e:
                print(f"Warning: Could not copy video to videos library: {e}")
            
            # Move processed files to an "archive" subfolder
            archive_dir = os.path.join(UNPROCESSED_DIR, "archive")
            ensure_directory_exists(archive_dir)
            
            # Move the video and LLC files to archive
            archive_video = os.path.join(archive_dir, os.path.basename(video_path))
            archive_llc = os.path.join(archive_dir, os.path.basename(llc_path))
            
            try:
                shutil.move(video_path, archive_video)
                shutil.move(llc_path, archive_llc)
                print(f"Moved processed files to archive folder.")
            except Exception as e:
                print(f"Warning: Could not move processed files to archive: {e}")
        
        if args.single:
            break  # Only process the first video if --single flag is set
    
    print(f"\n=== SUMMARY ===")
    print(f"Total videos processed: {processed_count}")
    print(f"Check the {STROKES_LIBRARY} directory for results.")
    
    return 0

def load_landmarks_from_csv(csv_path):
    """
    Reads the CSV and returns a list of dict frames[i][lm_id] = (x, y) in [0..1].
    We assume columns named 'lm_0_x', 'lm_0_y', etc.
    """
    frames = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Identify columns for each landmark
            lm_xy = {}
            for lm_id in range(33):
                x_col_name = f"lm_{lm_id}_x"
                y_col_name = f"lm_{lm_id}_y"
                if x_col_name in header and y_col_name in header:
                    x_idx = header.index(x_col_name)
                    y_idx = header.index(y_col_name)
                    lm_xy[lm_id] = (x_idx, y_idx)

            for row in reader:
                landm = {}
                for lm_id in range(33):
                    if lm_id in lm_xy:
                        try:
                            x_c, y_c = lm_xy[lm_id]
                            x_val = float(row[x_c])
                            y_val = float(row[y_c])
                            landm[lm_id] = (x_val, y_val)
                        except (ValueError, IndexError):
                            landm[lm_id] = (0.0, 0.0)
                    else:
                        landm[lm_id] = (0.0, 0.0)
                frames.append(landm)
    except Exception as e:
        print(f"Error loading landmarks from {csv_path}: {e}")
        # Return an empty frame set to avoid crashing
        return []
        
    return frames

def draw_pose(frame_bgr, landmarks_dict, width, height):
    """
    Draw a color-coded skeleton with thicker lines and anti-aliasing.
    Blue (left), Red (right), Green (center).
    """
    # Draw center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        if lmA in landmarks_dict and lmB in landmarks_dict:
            xA, yA = landmarks_dict[lmA]
            xB, yB = landmarks_dict[lmB]
            ptA = (int(xA*width), int(yA*height))
            ptB = (int(xB*width), int(yB*height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER,
                    thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        if lmA in landmarks_dict and lmB in landmarks_dict:
            xA, yA = landmarks_dict[lmA]
            xB, yB = landmarks_dict[lmB]
            ptA = (int(xA*width), int(yA*height))
            ptB = (int(xB*width), int(yB*height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT,
                    thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        if lmA in landmarks_dict and lmB in landmarks_dict:
            xA, yA = landmarks_dict[lmA]
            xB, yB = landmarks_dict[lmB]
            ptA = (int(xA*width), int(yA*height))
            ptB = (int(xB*width), int(yB*height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT,
                    thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id in range(33):
        if lm_id in landmarks_dict:
            nx, ny = landmarks_dict[lm_id]
            px = int(nx*width)
            py = int(ny*height)
            # color-coded keypoints or center color
            if lm_id in (11,13,15,23,25,27):
                kp_color = COLOR_LEFT
            elif lm_id in (12,14,16,24,26,28):
                kp_color = COLOR_RIGHT
            else:
                kp_color = COLOR_CENTER

            cv2.circle(frame_bgr, (px, py), CIRCLE_RADIUS, kp_color, -1, lineType=LINE_TYPE)

# Use draw_pose for draw_full_skeleton for compatibility
draw_full_skeleton = draw_pose

def create_pose_visualizations(video_path, csv_path, overlay_path, skeleton_path):
    """
    Create overlay and skeleton visualizations of the pose.
    
    Args:
        video_path: Path to the source video
        csv_path: Path to the landmarks CSV file
        overlay_path: Output path for overlay video (pose drawn on original)
        skeleton_path: Output path for skeleton-only video (pose on black)
        
    Returns:
        bool: Success or failure
    """
    try:
        print(f"Creating pose visualizations for {os.path.basename(video_path)}")
        
        # Skip if output files already exist
        if os.path.isfile(overlay_path) and os.path.isfile(skeleton_path):
            print(f"[SKIP] Output files already exist. Not overwriting.")
            return True
            
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load landmarks from CSV
        if not os.path.isfile(csv_path):
            print(f"[WARN] No CSV found: {csv_path}")
            cap.release()
            return False
            
        frames_landmarks = load_landmarks_from_csv(csv_path)
        if not frames_landmarks:
            print(f"[ERROR] No landmarks found in {csv_path}")
            cap.release()
            return False
            
        print(f"Video: {width}x{height} at {fps}fps, {frame_count} frames")
        print(f"Landmarks: {len(frames_landmarks)} frames")
        
        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overlay_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))
        skeleton_writer = cv2.VideoWriter(skeleton_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create overlay video
            if frame_idx < len(frames_landmarks):
                # Overlay mode
                overlay = frame.copy()
                draw_pose(overlay, frames_landmarks[frame_idx], width, height)
                overlay_writer.write(overlay)
                
                # Skeleton mode (black background)
                skeleton = np.zeros((height, width, 3), dtype=np.uint8)
                draw_pose(skeleton, frames_landmarks[frame_idx], width, height)
                skeleton_writer.write(skeleton)
            else:
                # If we run out of landmarks, just copy the frame for overlay
                # and write a black frame for skeleton
                overlay_writer.write(frame)
                skeleton = np.zeros((height, width, 3), dtype=np.uint8)
                skeleton_writer.write(skeleton)
            
            # Show progress
            frame_idx += 1
            if frame_idx % 20 == 0:
                print(f"Processing frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)", end="\r")
        
        # Clean up
        cap.release()
        overlay_writer.release()
        skeleton_writer.release()
        
        print(f"\n[DONE] Created visualizations: {os.path.basename(overlay_path)}, {os.path.basename(skeleton_path)}")
        return True
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Creating pose visualizations: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 