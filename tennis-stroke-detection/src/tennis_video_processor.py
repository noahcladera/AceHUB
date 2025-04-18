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
from collections import namedtuple
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

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

# Define MediaPipe Pose landmarks connection constants
# For color-coded skeleton drawing
POSE_CONNECTIONS_CENTER = [
    (11, 12),  # Shoulders
    (23, 24),  # Hips
    (11, 23),  # Left torso
    (12, 24),  # Right torso
]

POSE_CONNECTIONS_LEFT = [
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    (23, 25),  # Left hip to left knee
    (25, 27),  # Left knee to left ankle
    (27, 31),  # Left ankle to left foot
]

POSE_CONNECTIONS_RIGHT = [
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist 
    (24, 26),  # Right hip to right knee
    (26, 28),  # Right knee to right ankle
    (28, 32),  # Right ankle to right foot
]

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
    Extract MediaPipe pose data from a video and save as CSV.
    
    Args:
        video_path: Path to the video
        output_csv: Output CSV file path
        
    Returns:
        bool: Success or failure
    """
    if os.path.isfile(output_csv):
        print(f"[SKIP] {output_csv} exists. Not overwriting.")
        return True
        
    print(f"[INFO] Extracting pose data from {os.path.basename(video_path)} to {os.path.basename(output_csv)}")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # CSV header
    header = ["frame_index"]
    for lm in range(33):  # MediaPipe has 33 pose landmarks
        header.extend([f"lm_{lm}_x", f"lm_{lm}_y", f"lm_{lm}_z", f"lm_{lm}_vis"])
    header.extend(["right_elbow_angle", "left_elbow_angle"]) 
    
    # Process video
    all_frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Initialize row with zeros
        row = [frame_idx]
        for _ in range(33):
            row.extend([0.0, 0.0, 0.0, 0.0])  # x, y, z, visibility
        row.extend([0.0, 0.0])  # right_elbow_angle, left_elbow_angle
        
        # If pose detected, fill in the landmark data
        if results.pose_landmarks:
            for lm_idx, landmark in enumerate(results.pose_landmarks.landmark):
                idx = 1 + lm_idx * 4  # Position in row for this landmark
                row[idx] = landmark.x
                row[idx+1] = landmark.y
                row[idx+2] = landmark.z
                row[idx+3] = landmark.visibility
                
            # Calculate elbow angles if landmarks are available
            # Right elbow (shoulder-12, elbow-14, wrist-16)
            if all(lm_id in [12, 14, 16] for lm_id in [12, 14, 16]):
                shoulder = results.pose_landmarks.landmark[12]
                elbow = results.pose_landmarks.landmark[14]
                wrist = results.pose_landmarks.landmark[16]
                right_angle = calculate_angle(
                    shoulder.x, shoulder.y,
                    elbow.x, elbow.y,
                    wrist.x, wrist.y
                )
                row[-2] = right_angle
                
            # Left elbow (shoulder-11, elbow-13, wrist-15)
            if all(lm_id in [11, 13, 15] for lm_id in [11, 13, 15]):
                shoulder = results.pose_landmarks.landmark[11]
                elbow = results.pose_landmarks.landmark[13]
                wrist = results.pose_landmarks.landmark[15]
                left_angle = calculate_angle(
                    shoulder.x, shoulder.y,
                    elbow.x, elbow.y,
                    wrist.x, wrist.y
                )
                row[-1] = left_angle
        
        all_frames.append(row)
        
        # Update progress
        frame_idx += 1
        if frame_idx % 30 == 0:  # Update every 30 frames
            percent = (frame_idx / frame_count) * 100
            print(f"[INFO] Progress: {frame_idx}/{frame_count} frames ({percent:.1f}%)", end="\r")
    
    # Release resources
    cap.release()
    pose.close()
    
    # Save to CSV
    try:
        df = pd.DataFrame(all_frames, columns=header)
        df.to_csv(output_csv, index=False)
        print(f"\n[DONE] Saved {len(all_frames)} frames to {output_csv}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to save CSV: {e}")
        return False

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

def process_video(video_path, output_dir=None, stroke_extraction=True, create_overlay=True, create_skeleton=True):
    """
    Process a tennis video to extract pose data and optionally create skeleton videos.
    
    Args:
        video_path: Path to the input video
        output_dir: Output directory (defaults to same dir as video)
        stroke_extraction: Whether to extract stroke data
        create_overlay: Whether to create overlay video
        create_skeleton: Whether to create skeleton video
        
    Returns:
        bool: Success or failure
    """
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Define output paths
    csv_path = os.path.join(output_dir, f"{base_name}_pose.csv")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.mp4")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.mp4")
    
    # Extract pose data to CSV
    if stroke_extraction:
        extract_success = extract_pose_data(video_path, csv_path)
        if not extract_success:
            print(f"[ERROR] Failed to extract pose data from {video_path}")
            return False
    
    # Check if we need to create visualization videos
    if create_overlay or create_skeleton:
        # Read the CSV data
        try:
            frames_landmarks = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV file {csv_path}: {e}")
            return False
        
        # Create skeleton video
        if create_skeleton:
            skeleton_success = create_skeleton_video(video_path, skeleton_path, frames_landmarks)
            if not skeleton_success:
                print(f"[ERROR] Failed to create skeleton video")
        
        # Create overlay video
        if create_overlay:
            overlay_success = create_overlay_video(video_path, overlay_path, frames_landmarks)
            if not overlay_success:
                print(f"[ERROR] Failed to create overlay video")
    
    print(f"[DONE] Successfully processed {os.path.basename(video_path)}")
    return True

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
            result = create_pose_visualizations(
                clip_file_path, 
                csv_file_path,
                overlay_output,
                skeleton_output
            )
            
            if result is None:
                print(f"Failed to create visualizations for stroke_{stroke_id}")
            else:
                overlay_path, skeleton_path = result
                print(f"Created overlay video: {os.path.basename(overlay_path)}")
                print(f"Created skeleton video: {os.path.basename(skeleton_path)}")
        else:
            print(f"Missing clip or CSV file, cannot create visualizations for stroke_{stroke_id}")
        
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
        
        # Create an output directory for this video
        output_dir = os.path.join(DATA_PROCESSED_DIR, f"video_{current_video_id}")
        ensure_directory_exists(output_dir)
        
        # Define paths for CSVs
        data_csv = os.path.join(output_dir, f"video_{current_video_id}_data.csv")
        norm_csv = os.path.join(output_dir, f"video_{current_video_id}_normalized.csv")
        
        # First process the video to extract pose data
        print(f"Processing video {os.path.basename(video_path)} as video_{current_video_id}")
        success = process_video(video_path, output_dir)
        
        if success:
            # If processing succeeds, set up for the pipeline
            llc_processed = setup_for_pipeline(video_path, data_csv, norm_csv, llc_path, current_video_id)
            processed_count += 1
            
            # Mark as successful for the main loop
            success = True
        
        if success:
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

def draw_pretty_skeleton(frame=None, landmarks_row=None):
    """
    Draw a skeleton on a frame or on a black background using pose landmarks
    
    Args:
        frame: Optional frame to draw on. If None, a black frame is created
        landmarks_row: DataFrame row containing pose landmark data with columns 'lm_i_x', 'lm_i_y', 'lm_i_vis'
    
    Returns:
        Frame with skeleton drawn on it
    """
    # Create a black frame if none provided
    if frame is None:
        # Determine dimensions from landmarks
        max_x = max([landmarks_row[f'lm_{i}_x'] for i in range(33) if not pd.isna(landmarks_row[f'lm_{i}_x'])])
        max_y = max([landmarks_row[f'lm_{i}_y'] for i in range(33) if not pd.isna(landmarks_row[f'lm_{i}_y'])])
        
        # Default to 1280x720 if we can't determine from landmarks
        frame_width = int(max(1280, max_x * 1.1))
        frame_height = int(max(720, max_y * 1.1))
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    else:
        frame = frame.copy()  # Create a copy to avoid modifying the original

    # Colors for the skeleton (in BGR format)
    color_center = (255, 191, 0)    # Deep Sky Blue
    color_left = (0, 191, 255)      # Golden
    color_right = (255, 127, 80)    # Coral
    
    # Parameters for visualization
    line_thickness = 2
    circle_radius = 4
    min_visibility = 0.5  # Minimum visibility threshold
    
    # Draw center connections
    for connection in POSE_CONNECTIONS_CENTER:
        start_idx, end_idx = connection
        
        # Get landmark coordinates and visibility
        start_x = landmarks_row[f'lm_{start_idx}_x']
        start_y = landmarks_row[f'lm_{start_idx}_y'] 
        start_vis = landmarks_row[f'lm_{start_idx}_vis']
        
        end_x = landmarks_row[f'lm_{end_idx}_x']
        end_y = landmarks_row[f'lm_{end_idx}_y']
        end_vis = landmarks_row[f'lm_{end_idx}_vis']
        
        # Only draw if both landmarks are visible enough
        if start_vis > min_visibility and end_vis > min_visibility:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
            cv2.line(frame, start_point, end_point, color_center, line_thickness)
    
    # Draw left side connections
    for connection in POSE_CONNECTIONS_LEFT:
        start_idx, end_idx = connection
        
        # Get landmark coordinates and visibility
        start_x = landmarks_row[f'lm_{start_idx}_x']
        start_y = landmarks_row[f'lm_{start_idx}_y'] 
        start_vis = landmarks_row[f'lm_{start_idx}_vis']
        
        end_x = landmarks_row[f'lm_{end_idx}_x']
        end_y = landmarks_row[f'lm_{end_idx}_y']
        end_vis = landmarks_row[f'lm_{end_idx}_vis']
        
        # Only draw if both landmarks are visible enough
        if start_vis > min_visibility and end_vis > min_visibility:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
            cv2.line(frame, start_point, end_point, color_left, line_thickness)
    
    # Draw right side connections
    for connection in POSE_CONNECTIONS_RIGHT:
        start_idx, end_idx = connection
        
        # Get landmark coordinates and visibility
        start_x = landmarks_row[f'lm_{start_idx}_x']
        start_y = landmarks_row[f'lm_{start_idx}_y'] 
        start_vis = landmarks_row[f'lm_{start_idx}_vis']
        
        end_x = landmarks_row[f'lm_{end_idx}_x']
        end_y = landmarks_row[f'lm_{end_idx}_y']
        end_vis = landmarks_row[f'lm_{end_idx}_vis']
        
        # Only draw if both landmarks are visible enough
        if start_vis > min_visibility and end_vis > min_visibility:
            start_point = (int(start_x), int(start_y))
            end_point = (int(end_x), int(end_y))
            cv2.line(frame, start_point, end_point, color_right, line_thickness)
    
    # Draw landmarks
    for i in range(33):  # MediaPipe has 33 landmarks
        x = landmarks_row[f'lm_{i}_x']
        y = landmarks_row[f'lm_{i}_y']
        vis = landmarks_row[f'lm_{i}_vis']
        
        if vis > min_visibility:
            point = (int(x), int(y))
            
            # Choose color based on landmark position
            color = color_center
            # Check if this is a left-side landmark
            for connection in POSE_CONNECTIONS_LEFT:
                if i in connection:
                    color = color_left
                    break
            # Check if this is a right-side landmark
            for connection in POSE_CONNECTIONS_RIGHT:
                if i in connection:
                    color = color_right
                    break
                
            cv2.circle(frame, point, circle_radius, color, -1)  # -1 for filled circle
    
    return frame

def create_skeleton_video(input_video, output_path, landmarks_data):
    """
    Create a video with only the skeleton on a black background
    
    Args:
        input_video: Path to the input video file
        output_path: Path to save the output video
        landmarks_data: DataFrame containing pose landmarks data
    
    Returns:
        Path to the created skeleton video
    """
    try:
        # Open the video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Group landmarks by frame_index for easy lookup
        landmarks_by_frame = landmarks_data.groupby('frame_index')
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")
                
            # Check if we have landmarks for this frame
            if frame_idx in landmarks_by_frame.groups:
                # Get the landmarks for this frame
                frame_landmarks = landmarks_by_frame.get_group(frame_idx).iloc[0]
                
                # Create skeleton frame (on black background)
                skeleton_frame = draw_pretty_skeleton(None, frame_landmarks)
                
                # Write to output video
                out.write(skeleton_frame)
            else:
                # If no landmarks, write a black frame
                black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                out.write(black_frame)
                
            frame_idx += 1
            
        # Release resources
        cap.release()
        out.release()
        
        return output_path
        
    except Exception as e:
        print(f"Error creating skeleton video: {e}")
        return None

def create_overlay_video(input_video, output_path, landmarks_data):
    """
    Create a video with the skeleton overlaid on the original video
    
    Args:
        input_video: Path to the input video file
        output_path: Path to save the output video
        landmarks_data: DataFrame containing pose landmarks data
    
    Returns:
        Path to the created overlay video
    """
    try:
        # Open the video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Group landmarks by frame_index for easy lookup
        landmarks_by_frame = landmarks_data.groupby('frame_index')
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")
                
            # Check if we have landmarks for this frame
            if frame_idx in landmarks_by_frame.groups:
                # Get the landmarks for this frame
                frame_landmarks = landmarks_by_frame.get_group(frame_idx).iloc[0]
                
                # Create overlay by drawing skeleton on the original frame
                overlay_frame = draw_pretty_skeleton(frame, frame_landmarks)
                
                # Write to output video
                out.write(overlay_frame)
            else:
                # If no landmarks, just write the original frame
                out.write(frame)
                
            frame_idx += 1
            
        # Release resources
        cap.release()
        out.release()
        
        return output_path
        
    except Exception as e:
        print(f"Error creating overlay video: {e}")
        return None

def create_pose_visualizations(video_path, csv_path, overlay_path, skeleton_path):
    """
    Create pose visualization videos from a CSV file containing pose landmarks
    
    Args:
        video_path: Path to the input video
        csv_path: Path to the CSV file with pose landmarks
        overlay_path: Path to save the overlay video (skeleton on original video)
        skeleton_path: Path to save the skeleton-only video
        
    Returns:
        Tuple (overlay_path, skeleton_path) if successful, None if error
    """
    try:
        # Load landmarks data from CSV
        landmarks_data = pd.read_csv(csv_path)
        
        results = []
        
        # Create skeleton video if requested
        if skeleton_path:
            print(f"Creating skeleton video: {skeleton_path}")
            skeleton_result = create_skeleton_video(video_path, skeleton_path, landmarks_data)
            results.append(skeleton_result)
        
        # Create overlay video if requested
        if overlay_path:
            print(f"Creating overlay video: {overlay_path}")
            overlay_result = create_overlay_video(video_path, overlay_path, landmarks_data)
            results.append(overlay_result)
        
        # Check if all operations were successful
        if all(result is not None for result in results):
            return (overlay_path, skeleton_path)
        else:
            return None
            
    except Exception as e:
        print(f"Error creating pose visualizations: {e}")
        return None

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 