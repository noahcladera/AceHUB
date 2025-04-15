#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_video_unified.py

A unified script for processing tennis videos from start to finish in one step.

This script:
1. Takes a video from the unprocessed_videos directory 
2. Extracts pose landmarks with MediaPipe
3. Normalizes the data
4. Segments the video based on an LLC file
5. Creates clips, skeleton overlays, and 3D visualizations
6. Adds everything to the Strokes Library

Usage:
    python process_video_unified.py video_name.mp4 [--skip-steps STEPS]

Example:
    python process_video_unified.py my_tennis_video.mp4
    
    # Process but skip 3D visualization
    python process_video_unified.py my_tennis_video.mp4 --skip-steps 3d

    # Process a video with LLC file already created
    python process_video_unified.py my_tennis_video.mp4 --skip-steps llc

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
# MediaPipe Pose connections for skeleton drawing
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

# Color configuration
COLOR_LEFT = (255, 0, 0)  # Blue for left side
COLOR_RIGHT = (0, 0, 255)  # Red for right side
COLOR_CENTER = (0, 255, 0)  # Green for center

# Drawing configuration
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

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
    Convert MOV or other video formats to MP4 format.
    Returns True if conversion was successful.
    """
    try:
        print(f"Converting video format to MP4...")
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-c:v", "libx264", "-c:a", "aac", "-y", output_path],
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

def create_empty_llc_file(output_path, video_path):
    """
    Create an empty LLC file with instructions for manual annotation.
    """
    video_name = os.path.basename(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video to get duration")
        duration = "unknown"
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else "unknown"
        cap.release()
    
    with open(output_path, 'w') as f:
        f.write(f"# LLC file for {video_name}\n")
        f.write(f"# Video duration: {duration} seconds\n")
        f.write("# Format: start_time end_time stroke_type\n")
        f.write("# Example:\n")
        f.write("# 10.5 15.2 forehand\n")
        f.write("# 20.1 23.8 backhand\n")
        f.write("# 30.5 34.0 serve\n")
        f.write("\n# Add your annotations below:\n")
    
    print(f"[STEP 3/6] Created template LLC file at {output_path}")
    print(f"[ACTION REQUIRED] Please edit this file to annotate tennis strokes")
    print(f"Press Enter when you have finished annotating the LLC file...")
    input()

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
    
    print(f"[STEP 4/6] Setting up files for the main pipeline (video_{video_id})")
    print(f"[DONE] Files set up for pipeline as video_{video_id}")
    
    return target_llc_processed

def run_stroke_segmentation():
    """
    Run the stroke segmentation script to generate clips.
    """
    print(f"[STEP 5/6] Running stroke segmentation to generate clips")
    
    try:
        result = subprocess.run(
            ["python", "src/data/stroke_segmentation.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Stroke segmentation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running stroke segmentation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def get_strokes_for_video(video_id):
    """
    Find all strokes in the Strokes Library that came from the specified video.
    """
    strokes = []
    
    # Look through all stroke folders
    for stroke_dir in glob.glob(os.path.join(STROKES_LIBRARY, "stroke_*")):
        # Check source_info.txt to see if it's from our video
        source_info_path = os.path.join(stroke_dir, "source_info.txt")
        if os.path.exists(source_info_path):
            with open(source_info_path, 'r') as f:
                content = f.read()
                if f"Source Video: video_{video_id}" in content:
                    strokes.append(os.path.basename(stroke_dir))
    
    return sorted(strokes)

def create_3d_visualizations(video_id, strokes):
    """
    Create 3D visualizations of the normalized pose data.
    """
    print(f"[STEP 6/6] Creating 3D visualizations for {len(strokes)} strokes from video_{video_id}")
    
    # Process each stroke
    successful = 0
    for stroke in strokes:
        stroke_dir = os.path.join(STROKES_LIBRARY, stroke)
        
        # Check if we have the normalized CSV
        norm_csv = os.path.join(stroke_dir, "stroke_norm.csv")
        html_output = os.path.join(stroke_dir, "stroke_3d_viz.html")
        
        if not os.path.exists(norm_csv):
            print(f"[SKIP] Missing normalized CSV for {stroke}")
            continue
        
        # If output already exists, skip
        if os.path.exists(html_output):
            print(f"[SKIP] 3D visualization already exists for {stroke}")
            successful += 1
            continue
        
        print(f"Creating 3D visualization for {stroke}...")
        
        try:
            # Load the CSV data
            df = pd.read_csv(norm_csv)
            
            # Extract landmark coordinates
            x_coords, y_coords, z_coords, vis_scores = extract_landmark_data(df)
            
            # Create the visualization
            fig = create_animated_pose_figure(x_coords, y_coords, z_coords, vis_scores)
            
            # Save the HTML file
            fig.write_html(html_output)
            successful += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to create 3D visualization for {stroke}: {e}")
    
    print(f"[DONE] Created 3D visualizations for {successful}/{len(strokes)} strokes")
    return successful > 0

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
    Create an animated 3D plot of the pose sequence.
    """
    # Define MediaPipe Pose connections for the 3D visualization
    POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                       (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                       (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
                       (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                       (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for landmarks
    fig.add_trace(go.Scatter3d(
        x=x_coords[0],
        y=y_coords[0],
        z=z_coords[0],
        mode='markers',
        marker=dict(
            size=5,
            color=vis_scores[0],
            colorscale='Viridis',
            colorbar=dict(title='Visibility'),
            showscale=True
        ),
        name='Landmarks'
    ))
    
    # Add lines for connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        fig.add_trace(go.Scatter3d(
            x=[x_coords[0, start_idx], x_coords[0, end_idx]],
            y=[y_coords[0, start_idx], y_coords[0, end_idx]],
            z=[z_coords[0, start_idx], z_coords[0, end_idx]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # Create frames for animation
    frames = []
    for i in range(len(x_coords)):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x_coords[i],
                    y=y_coords[i],
                    z=z_coords[i],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=vis_scores[i],
                        colorscale='Viridis'
                    )
                )
            ] + [
                go.Scatter3d(
                    x=[x_coords[i, start_idx], x_coords[i, end_idx]],
                    y=[y_coords[i, start_idx], y_coords[i, end_idx]],
                    z=[z_coords[i, start_idx], z_coords[i, end_idx]],
                    mode='lines',
                    line=dict(color='red', width=2)
                )
                for start_idx, end_idx in POSE_CONNECTIONS
            ]
        )
        frames.append(frame)
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title='3D Pose Animation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]
            )]
        )],
        showlegend=True
    )
    
    return fig

def check_llc_file(llc_path):
    """
    Check if the LLC file has been edited with actual stroke annotations.
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

def main():
    parser = argparse.ArgumentParser(description="Process a tennis video from start to finish")
    parser.add_argument("video_name", help="Name of the video file in the unprocessed_videos directory")
    parser.add_argument("--skip-steps", help="Comma-separated list of steps to skip (pose,norm,llc,3d)")
    parser.add_argument("--video-id", type=int, help="Specify a video ID instead of auto-detecting")
    
    args = parser.parse_args()
    
    # Determine which steps to skip
    skip_steps = args.skip_steps.split(',') if args.skip_steps else []
    
    # Get full path to the video
    video_path = os.path.join(UNPROCESSED_DIR, args.video_name)
    
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Get video basename without extension
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(os.path.basename(video_path))[1].lower()
    
    # Create temporary work directory
    temp_dir = os.path.join(UNPROCESSED_DIR, f"{video_basename}_temp")
    ensure_directory_exists(temp_dir)
    
    # Convert MOV to MP4 if needed
    if video_ext.lower() != '.mp4':
        print(f"Video is in {video_ext} format, converting to MP4...")
        mp4_path = os.path.join(temp_dir, f"{video_basename}.mp4")
        if not convert_video_to_mp4(video_path, mp4_path):
            print("Failed to convert video. Make sure ffmpeg is installed.")
            return 1
        # Use the converted MP4 file for processing
        video_path = mp4_path
    
    # Define output paths
    data_csv = os.path.join(temp_dir, f"{video_basename}_data.csv")
    norm_csv = os.path.join(temp_dir, f"{video_basename}_normalized.csv")
    
    # Look for LLC file with both patterns: basename.llc or basename.ext.llc
    llc_path = os.path.join(UNPROCESSED_DIR, f"{video_basename}.llc")
    llc_path_with_ext = os.path.join(UNPROCESSED_DIR, f"{args.video_name}.llc")
    
    # Use the LLC file with extension if it exists and the basic one doesn't
    if not os.path.exists(llc_path) and os.path.exists(llc_path_with_ext):
        llc_path = llc_path_with_ext
        print(f"Found LLC file with full filename: {llc_path}")
    
    # Get video ID
    video_id = args.video_id if args.video_id is not None else get_next_video_id()
    
    # Step 1: Extract pose data
    if 'pose' not in skip_steps:
        if not extract_pose_data(video_path, data_csv):
            print("Error extracting pose data. Exiting.")
            return 1
    else:
        print("[SKIP] Skipping pose extraction")
    
    # Step 2: Normalize pose data
    if 'norm' not in skip_steps:
        if not normalize_pose_data(data_csv, norm_csv):
            print("Error normalizing pose data. Exiting.")
            return 1
    else:
        print("[SKIP] Skipping normalization")
    
    # Step 3: Create or check LLC file
    if 'llc' not in skip_steps:
        # First look for LLC file with full name including extension
        if not os.path.exists(llc_path) and not os.path.exists(llc_path_with_ext):
            # Create the LLC file with the same naming pattern as the input
            if args.video_name.lower().endswith('.mov') or args.video_name.lower().endswith('.mp4'):
                create_empty_llc_file(llc_path_with_ext, video_path)
            else:
                create_empty_llc_file(llc_path, video_path)
        elif not check_llc_file(llc_path):
            print(f"[WARNING] The LLC file exists but may not have valid annotations: {llc_path}")
            print("Press Enter to continue anyway, or Ctrl+C to abort and edit the LLC file...")
            input()
    else:
        print("[SKIP] Skipping LLC file creation/checking")
    
    # Step 4: Set up for pipeline
    target_llc = setup_for_pipeline(video_path, data_csv, norm_csv, llc_path, video_id)
    
    # Step 5: Run stroke segmentation
    if not run_stroke_segmentation():
        print("Error running stroke segmentation. Continuing anyway...")
    
    # Step 6: Create 3D visualizations
    strokes = get_strokes_for_video(video_id)
    if strokes:
        print(f"Found {len(strokes)} strokes from video_{video_id}")
        
        if '3d' not in skip_steps:
            create_3d_visualizations(video_id, strokes)
        else:
            print("[SKIP] Skipping 3D visualization creation")
    else:
        print(f"No strokes found for video_{video_id}. Check the LLC file for proper annotations.")
    
    # Clean up temporary files
    print(f"Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Processed {video_basename} as video_{video_id}")
    print(f"Generated {len(strokes)} stroke clips in the Strokes Library")
    print(f"Results are available in {STROKES_LIBRARY}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 