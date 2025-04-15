#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
personal_pre_processing.py

All-in-one pre-processing script for adding a new personal video to the tennis analysis pipeline.
This script handles:
1. Extracting pose data from a video using MediaPipe
2. Normalizing the pose data
3. Setting up the directory structure for the main pipeline

Usage:
    python personal_pre_processing.py input_video.mp4 [--output-dir OUTPUT_DIR] [--next-id VIDEO_ID]

Example:
    python personal_pre_processing.py my_tennis_class.mp4 --next-id 75

Dependencies:
    pip install opencv-python mediapipe numpy pandas
"""

import os
import sys
import shutil
import argparse
import csv
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# Constants
DEFAULT_OUTPUT_DIR = "data/personal"
PERSONAL_DIR = os.path.join("data", "personal")
PROCESSED_DIR = os.path.join("data", "processed")
VIDEOS_DIR = "videos"

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def get_next_video_id():
    """Find the next available video ID by checking existing folders"""
    try:
        # Check videos directory first
        video_dirs = [d for d in os.listdir(VIDEOS_DIR) if d.startswith("video_") and os.path.isdir(os.path.join(VIDEOS_DIR, d))]
        
        # If videos directory doesn't exist, check processed directory
        if not video_dirs and os.path.exists(PROCESSED_DIR):
            video_dirs = [d for d in os.listdir(PROCESSED_DIR) if d.startswith("video_") and os.path.isdir(os.path.join(PROCESSED_DIR, d))]
        
        # Extract IDs and find max
        if video_dirs:
            video_ids = [int(d.split("_")[1]) for d in video_dirs]
            return max(video_ids) + 1
    except Exception as e:
        print(f"Error finding next video ID: {e}")
    
    # Default if no videos found or error
    return 1

def extract_pose_data(video_path, output_csv):
    """
    Extract pose landmarks from video using MediaPipe.
    """
    print(f"[STEP 1/3] Extracting pose data from {os.path.basename(video_path)}")
    
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
                # Right side: 12=Rshoulder, 14=Relbow, 16=Rwrist
                r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                # Left side: 11=Lshoulder, 13=Lelbow, 15=LWrist
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

def calculate_angle(ax, ay, bx, by, cx, cy):
    """
    Calculate the angle at point B formed by A->B->C in 2D space.
    """
    import math
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
    print(f"[STEP 2/3] Normalizing pose data")
    
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
    
    print(f"[INFO] Created template LLC file at {output_path}")
    print(f"[INFO] Please open this file and add stroke annotations manually")

def setup_for_pipeline(video_path, data_csv, norm_csv, llc_path, video_id):
    """
    Set up the files in the proper structure for the main pipeline.
    """
    print(f"[STEP 3/3] Setting up files for the main pipeline (video_{video_id})")
    
    # Create the necessary directories
    video_dir_processed = os.path.join(PROCESSED_DIR, f"video_{video_id}")
    video_dir_videos = os.path.join(VIDEOS_DIR, f"video_{video_id}")
    
    ensure_directory_exists(video_dir_processed)
    ensure_directory_exists(video_dir_videos)
    
    # Define target files
    target_video_processed = os.path.join(video_dir_processed, f"video_{video_id}.mp4")
    target_data_processed = os.path.join(video_dir_processed, f"video_{video_id}_data.csv")
    target_norm_processed = os.path.join(video_dir_processed, f"video_{video_id}_normalized.csv")
    target_llc_processed = os.path.join(video_dir_processed, f"video_{video_id}.llc")
    
    # Copy files to processed directory
    shutil.copy2(video_path, target_video_processed)
    shutil.copy2(data_csv, target_data_processed)
    shutil.copy2(norm_csv, target_norm_processed)
    shutil.copy2(llc_path, target_llc_processed)
    
    # Copy files to videos directory
    shutil.copy2(video_path, os.path.join(video_dir_videos, f"video_{video_id}.mp4"))
    shutil.copy2(norm_csv, os.path.join(video_dir_videos, f"video_{video_id}_normalized.csv"))
    shutil.copy2(llc_path, os.path.join(video_dir_videos, f"video_{video_id}.llc"))
    
    # Create status file in videos directory
    status_path = os.path.join(video_dir_videos, "status.txt")
    with open(status_path, 'w') as f:
        f.write("has_video: True\n")
        f.write("has_normalized_csv: True\n")
        f.write("has_data_csv: False\n")  # We don't copy the raw data CSV to videos/
        f.write("has_llc: True\n")
        f.write("has_clips: False\n")
        f.write("is_ready_for_clipping: True\n")
        f.write("is_fully_processed: False\n")
    
    print(f"[DONE] Files set up for pipeline as video_{video_id}")
    print(f"Next steps:")
    print(f"1. Manually annotate the LLC file at: {target_llc_processed}")
    print(f"2. Run the personal_post_processing.py script to generate clips")

def main():
    parser = argparse.ArgumentParser(description="Process a personal tennis video for analysis")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for processed files")
    parser.add_argument("--next-id", type=int, help="Next video ID to use (if not specified, will be auto-detected)")
    
    args = parser.parse_args()
    
    # Get video filename without extension
    video_basename = os.path.basename(args.video_path)
    video_name = os.path.splitext(video_basename)[0]
    
    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output_dir)
    
    # Define output paths
    data_csv = os.path.join(args.output_dir, f"{video_name}_data.csv")
    norm_csv = os.path.join(args.output_dir, f"{video_name}_normalized.csv")
    llc_path = os.path.join(args.output_dir, f"{video_name}.llc")
    
    # Step 1: Extract pose data
    extract_pose_data(args.video_path, data_csv)
    
    # Step 2: Normalize pose data
    normalize_pose_data(data_csv, norm_csv)
    
    # Create LLC file template
    create_empty_llc_file(llc_path, args.video_path)
    
    # Get next video ID (auto-detect or use provided)
    video_id = args.next_id if args.next_id is not None else get_next_video_id()
    
    # Step 3: Set up for pipeline
    setup_for_pipeline(args.video_path, data_csv, norm_csv, llc_path, video_id)
    
    print("\n=== PERSONAL PRE-PROCESSING COMPLETE ===")
    print(f"Your video has been processed with ID: video_{video_id}")

if __name__ == "__main__":
    main() 