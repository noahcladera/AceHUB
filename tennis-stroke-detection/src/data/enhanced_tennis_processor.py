#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
enhanced_tennis_processor.py

An improved version of the tennis video processor with enhanced pose detection and visualization.
This version incorporates the successful approach from process_video_skeleton.py while maintaining
the full tennis video processing pipeline.

Usage:
    python enhanced_tennis_processor.py [--video-path VIDEO_PATH] [--single]

Example:
    python enhanced_tennis_processor.py --video-path unprocessed_videos/DSC_5228.MOV
    python enhanced_tennis_processor.py --single  # Process only the first video found

Dependencies:
    pip install opencv-python mediapipe numpy pandas
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
import time
import math
import json
import re

# Directory constants
UNPROCESSED_DIR = "unprocessed_videos"
VIDEOS_DIR = "videos"
STROKES_LIBRARY = "Strokes_Library"
DATA_PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "Output"

# Visualization constants
# Color configuration
COLOR_LEFT = (255, 0, 0)  # Blue in BGR
COLOR_RIGHT = (0, 0, 255)  # Red in BGR
COLOR_CENTER = (0, 255, 0)  # Green in BGR

# Define pose connections for color-coding
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

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

def draw_pretty_skeleton(frame_bgr, landmarks, width, height):
    """
    Draws a color-coded skeleton with thicker lines and anti-aliasing.
    Blue (left), Red (right), Green (center).
    
    This is the enhanced version from process_video_skeleton.py.
    """
    # Center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        if lmA in landmarks and lmB in landmarks:
            xA, yA = landmarks[lmA]
            xB, yB = landmarks[lmB]
            ptA = (int(xA * width), int(yA * height))
            ptB = (int(xB * width), int(yB * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        if lmA in landmarks and lmB in landmarks:
            xA, yA = landmarks[lmA]
            xB, yB = landmarks[lmB]
            ptA = (int(xA * width), int(yA * height))
            ptB = (int(xB * width), int(yB * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        if lmA in landmarks and lmB in landmarks:
            xA, yA = landmarks[lmA]
            xB, yB = landmarks[lmB]
            ptA = (int(xA * width), int(yA * height))
            ptB = (int(xB * width), int(yB * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id, (nx, ny) in landmarks.items():
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

def process_video_with_mediapipe(video_path, output_prefix=None):
    """
    Process a video with MediaPipe Pose and create output videos.
    This is the enhanced implementation from process_video_skeleton.py.
    
    Args:
        video_path: Path to the input video
        output_prefix: Prefix for output files (default: video filename without extension)
        
    Returns:
        tuple: (skeleton_output, overlay_output, landmarks_dict_list, csv_output)
    """
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"\n=== PROCESSING VIDEO: {os.path.basename(video_path)} ===")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Use full complexity for better results
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None, None, None, None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video: {width}x{height} at {fps} FPS, {total_frames} frames")
    
    # Create output directory if it doesn't exist
    ensure_directory_exists(OUTPUT_DIR)
    
    # Setup output video writers
    skeleton_output = f"{OUTPUT_DIR}/{output_prefix}_skeleton.mp4"
    overlay_output = f"{OUTPUT_DIR}/{output_prefix}_overlay.mp4"
    csv_output = f"{OUTPUT_DIR}/{output_prefix}_data.csv"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_writer = cv2.VideoWriter(skeleton_output, fourcc, fps, (width, height))
    overlay_writer = cv2.VideoWriter(overlay_output, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing video and creating:")
    print(f"       - {skeleton_output}")
    print(f"       - {overlay_output}")
    print(f"       - {csv_output}")
    
    # Prepare CSV file
    with open(csv_output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Build header
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        # Add angles
        header += ["right_elbow_angle", "left_elbow_angle", "stroke_label"]
        writer.writerow(header)
    
    # Process frame by frame
    frame_count = 0
    landmarks_dict_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # If pose detected, extract landmarks
        if results.pose_landmarks:
            # Convert landmarks to dictionary format {lm_id: (x, y)}
            landmarks_dict = {}
            for lm_id, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_dict[lm_id] = (landmark.x, landmark.y)
            
            landmarks_dict_list.append(landmarks_dict)
            
            # Create black background for skeleton-only video
            skeleton_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw skeleton on black background
            draw_pretty_skeleton(skeleton_frame, landmarks_dict, width, height)
            
            # Draw skeleton on original frame (for overlay)
            overlay_frame = frame.copy()
            draw_pretty_skeleton(overlay_frame, landmarks_dict, width, height)
            
            # Write frames to output videos
            skeleton_writer.write(skeleton_frame)
            overlay_writer.write(overlay_frame)
            
            # Save to CSV
            with open(csv_output, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                row = [frame_count]
                
                # Add all landmark data
                for lm_id in range(33):
                    if lm_id in landmarks_dict:
                        x, y = landmarks_dict[lm_id]
                        # For z and visibility, use dummy values if not available
                        z = results.pose_landmarks.landmark[lm_id].z if hasattr(results.pose_landmarks.landmark[lm_id], 'z') else 0.0
                        vis = results.pose_landmarks.landmark[lm_id].visibility if hasattr(results.pose_landmarks.landmark[lm_id], 'visibility') else 1.0
                        row.extend([x, y, z, vis])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0.0])
                
                # Calculate elbow angles
                if 11 in landmarks_dict and 13 in landmarks_dict and 15 in landmarks_dict:
                    l_shoulder = results.pose_landmarks.landmark[11]
                    l_elbow = results.pose_landmarks.landmark[13]
                    l_wrist = results.pose_landmarks.landmark[15]
                    l_angle = calculate_angle(l_shoulder.x, l_shoulder.y, l_elbow.x, l_elbow.y, l_wrist.x, l_wrist.y)
                else:
                    l_angle = 0.0
                    
                if 12 in landmarks_dict and 14 in landmarks_dict and 16 in landmarks_dict:
                    r_shoulder = results.pose_landmarks.landmark[12]
                    r_elbow = results.pose_landmarks.landmark[14]
                    r_wrist = results.pose_landmarks.landmark[16]
                    r_angle = calculate_angle(r_shoulder.x, r_shoulder.y, r_elbow.x, r_elbow.y, r_wrist.x, r_wrist.y)
                else:
                    r_angle = 0.0
                
                # Add angles and default stroke label (1)
                row.extend([r_angle, l_angle, 1])
                writer.writerow(row)
        else:
            # No pose detected, just write original frame to overlay 
            # and black frame to skeleton
            skeleton_writer.write(np.zeros((height, width, 3), dtype=np.uint8))
            overlay_writer.write(frame)
            landmarks_dict_list.append({})  # Empty dict for this frame
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            percent = (frame_count / total_frames) * 100
            print(f"[INFO] Progress: {frame_count}/{total_frames} frames ({percent:.1f}%)")
    
    # Release resources
    cap.release()
    skeleton_writer.release()
    overlay_writer.release()
    pose.close()
    
    print(f"[DONE] Successfully processed video")
    print(f"       - Skeleton video: {skeleton_output}")
    print(f"       - Overlay video: {overlay_output}")
    print(f"       - Landmark data: {csv_output}")
    
    return skeleton_output, overlay_output, landmarks_dict_list, csv_output

def normalize_pose_data(input_csv, output_csv):
    """
    Normalize the pose data to make it invariant to position and scale.
    """
    print(f"[STEP 2] Normalizing pose data")
    
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
        
        # Include stroke label if it exists
        if 'stroke_label' in df.columns:
            norm_df['stroke_label'] = df['stroke_label']
        else:
            norm_df['stroke_label'] = 1  # Default stroke label
        
        # Save normalized data
        norm_df.to_csv(output_csv, index=False)
        print(f"[DONE] Normalized data saved to {output_csv}")
        
    except Exception as e:
        print(f"Error during normalization: {e}")
        return False
    
    return True

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
    Set up the files for the main pipeline by copying to appropriate directories
    """
    print(f"[STEP 3] Setting up files for video_{video_id}")
    
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
        print(f"[INFO] Copied video to {target_video}")
    else:
        print(f"[WARN] Video file not found: {video_path}")
        
    if os.path.exists(data_csv):
        shutil.copy2(data_csv, target_data_csv)
        print(f"[INFO] Copied data CSV to {target_data_csv}")
    else:
        print(f"[WARN] Data CSV not found: {data_csv}")
        
    if os.path.exists(norm_csv):
        shutil.copy2(norm_csv, target_norm_csv)
        print(f"[INFO] Copied normalized CSV to {target_norm_csv}")
    else:
        print(f"[WARN] Normalized CSV not found: {norm_csv}")
        
    if os.path.exists(llc_path):
        shutil.copy2(llc_path, target_llc_processed)
        print(f"[INFO] Copied LLC file to {target_llc_processed}")
    else:
        print(f"[WARN] LLC file not found: {llc_path}")
    
    # Copy files to videos/video_X/
    video_file = os.path.join(video_dir_videos, f"video_{video_id}.mp4")
    if os.path.exists(video_path):
        shutil.copy2(video_path, video_file)
        print(f"[INFO] Copied video to {video_file}")
        
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
    
    print(f"[DONE] Files set up for pipeline as video_{video_id}")
    return target_llc_processed

def run_stroke_segmentation():
    """
    Run the stroke segmentation script to generate clips.
    """
    print(f"[STEP 4] Running stroke segmentation to generate clips")
    
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
            
        if result.returncode != 0:
            print(f"Stroke segmentation returned non-zero exit code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            print("Continuing despite errors...")
        
        return True
    except Exception as e:
        import traceback
        print(f"Exception running stroke segmentation: {e}")
        print(traceback.format_exc())
        print("Continuing despite errors...")
        return False

def process_single_video(video_path, llc_path=None, video_id=None):
    """
    Process a single video with enhanced pose detection and visualization
    
    Args:
        video_path: Path to the video file
        llc_path: Optional path to the LLC file
        video_id: Optional video ID
        
    Returns:
        bool: Success or failure
    """
    try:
        # Get video ID if not provided
        if video_id is None:
            video_id = get_next_video_id()
        
        # Get output prefix
        output_prefix = f"video_{video_id}"
        
        print(f"\n=== PROCESSING VIDEO AS {output_prefix} ===")
        
        # Step 1: Process video with MediaPipe
        print(f"[STEP 1] Processing video with MediaPipe")
        skeleton_output, overlay_output, landmarks_dict_list, csv_output = process_video_with_mediapipe(
            video_path, output_prefix)
        
        if not csv_output or not os.path.exists(csv_output):
            print(f"[ERROR] Failed to process video or create CSV output")
            return False
        
        # Step 2: Normalize data
        norm_csv = f"{OUTPUT_DIR}/{output_prefix}_normalized.csv"
        normalize_pose_data(csv_output, norm_csv)
        
        # If LLC path is provided, setup for pipeline
        if llc_path and os.path.exists(llc_path):
            # Step 3: Set up for pipeline
            setup_for_pipeline(video_path, csv_output, norm_csv, llc_path, video_id)
            
            # Step 4: Run stroke segmentation if llc file exists
            if check_llc_file(llc_path):
                print(f"[INFO] Valid LLC file found, running stroke segmentation")
                run_stroke_segmentation()
            else:
                print(f"[WARN] LLC file has no valid stroke annotations, skipping segmentation")
        else:
            print(f"[INFO] No LLC file provided, skipping pipeline setup and segmentation")
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Processed {os.path.basename(video_path)} as {output_prefix}")
        print(f"Results are available in the {OUTPUT_DIR} directory")
        
        return True
    except Exception as e:
        import traceback
        print(f"[ERROR] Processing video: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main function to process videos"""
    parser = argparse.ArgumentParser(description="Enhanced Tennis Video Processor")
    parser.add_argument("--video-path", help="Path to a specific video to process")
    parser.add_argument("--llc-path", help="Path to a corresponding LLC file")
    parser.add_argument("--video-id", type=int, help="Specify a starting video ID")
    parser.add_argument("--single", action="store_true", help="Process only the first video found")
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    ensure_directory_exists(UNPROCESSED_DIR)
    ensure_directory_exists(VIDEOS_DIR)
    ensure_directory_exists(STROKES_LIBRARY)
    ensure_directory_exists(DATA_PROCESSED_DIR)
    ensure_directory_exists(OUTPUT_DIR)
    
    # If specific video path is provided, process just that video
    if args.video_path:
        if os.path.exists(args.video_path):
            llc_path = args.llc_path
            
            # If no LLC path is provided, try to find it
            if not llc_path:
                base_name = os.path.splitext(args.video_path)[0]
                potential_llc = f"{base_name}.llc"
                if os.path.exists(potential_llc):
                    llc_path = potential_llc
                    print(f"[INFO] Found LLC file: {llc_path}")
                
                potential_llc_full = f"{args.video_path}.llc"
                if os.path.exists(potential_llc_full):
                    llc_path = potential_llc_full
                    print(f"[INFO] Found LLC file: {llc_path}")
            
            video_id = args.video_id if args.video_id is not None else get_next_video_id()
            success = process_single_video(args.video_path, llc_path, video_id)
            return 0 if success else 1
        else:
            print(f"[ERROR] Video file not found: {args.video_path}")
            return 1
    
    # Otherwise, look for video+LLC pairs in the unprocessed directory
    video_llc_pairs = get_video_and_llc_paths()
    
    if not video_llc_pairs:
        print("No video+LLC pairs found in the unprocessed_videos directory.")
        print("Please specify a video path with --video-path or add videos and LLCs to the unprocessed_videos directory.")
        return 1
    
    print(f"Found {len(video_llc_pairs)} video+LLC pairs to process.")
    
    # Process videos
    video_id = args.video_id if args.video_id is not None else get_next_video_id()
    processed_count = 0
    
    for video_path, llc_path in video_llc_pairs:
        current_video_id = video_id + processed_count
        success = process_single_video(video_path, llc_path, current_video_id)
        
        if success:
            processed_count += 1
            
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
    print(f"Results are available in the {OUTPUT_DIR} directory.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 