#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merged_tennis_processor.py

A complete end-to-end script that combines the best of tennis_video_processor.py 
and enhanced_tennis_processor.py, creating high-quality visualizations and proper segmentation.

This script:
1. Takes a video and its corresponding LosslessCut (LLC) file from unprocessed_videos directory
2. Extracts pose landmarks with MediaPipe using enhanced visualization
3. Creates high-quality skeleton and overlay videos
4. Processes segmentation based on the LLC file for all three video types
5. Adds everything to the Strokes Library with proper organization

Usage:
    python merged_tennis_processor.py [--video-id VIDEO_ID] [--single]

Example:
    python merged_tennis_processor.py
    python merged_tennis_processor.py --single

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
from scipy.interpolate import interp1d

# Directory constants
UNPROCESSED_DIR = "unprocessed_videos"
VIDEOS_DIR = "videos"
STROKES_LIBRARY = "Strokes_Library"
OUTPUT_DIR = "Output"  # Temporary directory for processing

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
        print(f"[ERROR] Unprocessed videos directory not found: {UNPROCESSED_DIR}")
        return []
    
    try:
        video_files = []
        llc_files = []
        
        for file in os.listdir(UNPROCESSED_DIR):
            file_path = os.path.join(UNPROCESSED_DIR, file)
            
            # Skip directories and hidden files
            if os.path.isdir(file_path) or file.startswith('.'):
                continue
                
            lower_file = file.lower()
            if lower_file.endswith(('.mp4', '.mov', '.avi')):
                video_files.append(file)
            elif lower_file.endswith('.llc'):
                llc_files.append(file)
                
        # Match videos with LLC files
        video_llc_pairs = []
        
        for video_file in video_files:
            video_path = os.path.join(UNPROCESSED_DIR, video_file)
            base_name = os.path.splitext(video_file)[0]
            
            # Look for LLC file with same base name
            llc_file = f"{base_name}.llc"
            llc_path = os.path.join(UNPROCESSED_DIR, llc_file)
            
            if llc_file in llc_files:
                video_llc_pairs.append((video_path, llc_path))
            else:
                # Also check for LLC with full filename
                full_llc = f"{video_file}.llc"
                full_llc_path = os.path.join(UNPROCESSED_DIR, full_llc)
                
                if full_llc in llc_files:
                    video_llc_pairs.append((video_path, full_llc_path))
                else:
                    print(f"[WARN] No LLC file found for {video_file}, skipping.")
        
        return video_llc_pairs
        
    except Exception as e:
        print(f"[ERROR] Failed to scan unprocessed videos directory: {e}")
        return []

def get_next_video_id():
    """Find the next available video ID by checking existing folders"""
    try:
        video_dirs = []
        
        # Check videos directory
        if os.path.exists(VIDEOS_DIR):
            video_dirs.extend([d for d in os.listdir(VIDEOS_DIR) 
                              if d.startswith("video_") and os.path.isdir(os.path.join(VIDEOS_DIR, d))])
        
        # Check processed directory
        if os.path.exists(OUTPUT_DIR):
            video_dirs.extend([d for d in os.listdir(OUTPUT_DIR) 
                              if d.startswith("video_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))])
        
        # Extract IDs and find max
        if video_dirs:
            video_ids = [int(d.split("_")[1]) for d in video_dirs]
            return max(video_ids) + 1
    except Exception as e:
        print(f"Error finding next video ID: {e}")
    
    # Default if no videos found or error
    return 1

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

def draw_pretty_skeleton(frame_bgr, landmarks, width, height):
    """
    Draws a color-coded skeleton with thicker lines and anti-aliasing.
    Blue (left), Red (right), Green (center).
    
    This is the enhanced version from enhanced_tennis_processor.py.
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

def process_video_with_mediapipe(video_path, output_prefix=None, video_id=None):
    """
    Process a video with MediaPipe Pose and create output videos.
    This is the enhanced implementation from enhanced_tennis_processor.py.
    
    Args:
        video_path: Path to the input video
        output_prefix: Prefix for output files (default: video filename without extension)
        video_id: Video ID for direct output to videos folder
        
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
    
    # Set up output paths 
    # If video_id is provided, output directly to videos folder
    if video_id is not None:
        videos_dir_path = os.path.join(VIDEOS_DIR, f"video_{video_id}")
        ensure_directory_exists(videos_dir_path)
        skeleton_output = os.path.join(videos_dir_path, f"video_{video_id}_skeleton.mp4")
        overlay_output = os.path.join(videos_dir_path, f"video_{video_id}_overlay.mp4")
        csv_output = os.path.join(videos_dir_path, f"video_{video_id}_data.csv")
    else:
        # Output to temporary location
        ensure_directory_exists(OUTPUT_DIR)
        skeleton_output = f"{OUTPUT_DIR}/{output_prefix}_skeleton.mp4"
        overlay_output = f"{OUTPUT_DIR}/{output_prefix}_overlay.mp4"
        csv_output = f"{OUTPUT_DIR}/{output_prefix}_data.csv"
    
    print(f"[INFO] Processing video and creating:")
    print(f"       - {skeleton_output}")
    print(f"       - {overlay_output}")
    print(f"       - {csv_output}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_writer = cv2.VideoWriter(skeleton_output, fourcc, fps, (width, height))
    overlay_writer = cv2.VideoWriter(overlay_output, fourcc, fps, (width, height))
    
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
    
    try:
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
                
                # Write empty row to CSV
                with open(csv_output, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    row = [frame_count]
                    for i in range(33):
                        row.extend([0.0, 0.0, 0.0, 0.0])  # x, y, z, visibility
                    row.extend([0.0, 0.0, 1])  # right_elbow_angle, left_elbow_angle, stroke_label
                    writer.writerow(row)
            
            # Update progress
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                percent = (frame_count / total_frames) * 100
                print(f"[INFO] Progress: {frame_count}/{total_frames} frames ({percent:.1f}%)")
    except Exception as e:
        print(f"[ERROR] Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
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

def normalize_pose_data(input_csv, output_csv=None, video_id=None):
    """
    Normalize the pose data to make it invariant to position and scale.
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Path to the output CSV file (if None, will generate based on video_id)
        video_id: Video ID to use for direct output path
    
    Returns:
        Path to the normalized CSV file
    """
    print(f"[STEP 2] Normalizing pose data")
    
    # Set up output path
    if output_csv is None:
        if video_id is not None:
            videos_dir_path = os.path.join(VIDEOS_DIR, f"video_{video_id}")
            ensure_directory_exists(videos_dir_path)
            output_csv = os.path.join(videos_dir_path, f"video_{video_id}_normalized.csv")
        else:
            # Default output path
            base_dir = os.path.dirname(input_csv)
            base_name = os.path.basename(input_csv).replace("_data.csv", "_normalized.csv")
            output_csv = os.path.join(base_dir, base_name)
    
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
        print(f"[ERROR] Error during normalization: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return output_csv

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

# -------------------------------------------------------------------------
# INTEGRATED STROKE SEGMENTATION - Feature Engineering, Clip Generation,
# Time Normalization, all in one place
# -------------------------------------------------------------------------

# Constants for segmentation
FPS = 30
RESAMPLED_FRAMES = 120

def convert_llc_to_valid_json(text):
    """
    Convert LLC file text to valid JSON by:
    1. Adding quotes around property names
    2. Replacing single quotes with double quotes
    3. Removing trailing commas in objects and arrays
    """
    # Add quotes to property names (that aren't already quoted)
    text = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
    
    # Replace single quotes with double quotes
    text = text.replace("'", '"')
    
    # Remove trailing commas in objects
    text = re.sub(r',(\s*)}', r'\1}', text)
    
    # Remove trailing commas in arrays
    text = re.sub(r',(\s*)]', r'\1]', text)
    
    return text

def load_segments_from_llc(llc_path):
    """
    Load segments from an LLC file and INVERT them to get stroke segments.
    
    In LosslessCut, "cutSegments" are the parts to CUT OUT, not keep.
    We need to invert these to get the parts BETWEEN them, which are the actual strokes.
    
    Returns:
        list of (start_frame, end_frame) tuples for the inverted segments (actual strokes)
    """
    if not os.path.exists(llc_path):
        print(f"[ERROR] LLC file not found: {llc_path}")
        return []
    
    with open(llc_path, 'r', encoding='utf-8') as f:
        llc_content = f.read().strip()
    
    # Check if the LLC content is in JSON format
    if llc_content.startswith("{"):
        try:
            # First try to clean up to valid JSON
            cleaned_content = convert_llc_to_valid_json(llc_content)
            
            try:
                # Try to parse the cleaned content
                data = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse LLC file as JSON after cleanup: {e}")
                print(f"Cleaned content: {cleaned_content[:100]}...")
                return []
            
            # Extract cut segments from JSON data - these are parts to remove
            cut_segments = []
            
            # Look for cutSegments in the JSON (most likely format)
            if "cutSegments" in data and isinstance(data["cutSegments"], list):
                print(f"[INFO] Found {len(data['cutSegments'])} cutSegments in LLC file")
                for segment in data["cutSegments"]:
                    if isinstance(segment, dict):
                        # Handle time in seconds (convert to frames)
                        if "start" in segment and "end" in segment:
                            start_time = float(segment["start"])
                            end_time = float(segment["end"])
                            start_frame = int(start_time * FPS)
                            end_frame = int(end_time * FPS)
                            cut_segments.append((start_frame, end_frame))
            
            # If no cut segments found, look for other formats
            if not cut_segments:
                # Look for segments directly in the JSON
                for key in ["segments", "cuts", "strokes"]:
                    if key in data and isinstance(data[key], list):
                        for segment in data[key]:
                            if isinstance(segment, dict):
                                start_frame = segment.get("start", 0)
                                end_frame = segment.get("end", 0)
                                if start_frame < end_frame:
                                    cut_segments.append((int(start_frame), int(end_frame)))
            
            if not cut_segments:
                print(f"[WARN] No cut segments found in JSON LLC file: {llc_path}")
                return []
            
            # INVERT the segments to get the parts BETWEEN the cut segments
            # These are the actual stroke segments we want to keep
            
            # Sort cut segments by start time
            cut_segments.sort(key=lambda x: x[0])
            
            # Determine video duration (approximate from last segment or use a large value)
            video_duration_frames = cut_segments[-1][1] + 300  # Add buffer frames
            
            # Create stroke segments from spaces between cut segments
            stroke_segments = []
            prev_end = 0
            
            for start, end in cut_segments:
                if start > prev_end:
                    # This is a part to keep (between cuts)
                    stroke_segments.append((prev_end, start - 1))
                prev_end = end + 1
            
            # Add final segment if needed
            if prev_end < video_duration_frames:
                stroke_segments.append((prev_end, video_duration_frames))
            
            print(f"[INFO] Inverted {len(cut_segments)} cut segments into {len(stroke_segments)} stroke segments")
            return stroke_segments
            
        except Exception as e:
            print(f"[ERROR] Failed to process LLC file: {e}")
            return []
    else:
        # Assume simple text format with one segment per line
        # For text format, assume these are already stroke segments, not cuts
        segments = []
        lines = llc_content.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    parts = line.split()
                    if len(parts) >= 2:  # At least start_time and end_time
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        start_frame = int(start_time * FPS)
                        end_frame = int(end_time * FPS)
                        segments.append((start_frame, end_frame))
                except (ValueError, IndexError) as e:
                    print(f"[WARN] Error parsing line in LLC file: {line} - {e}")
        
        if not segments:
            print(f"[WARN] No segments found in text LLC file: {llc_path}")
        
        return segments

def create_frame_labels(pose_csv_path, llc_path, output_csv_path):
    """
    Create a labeled CSV file by adding frame-level stroke labels based on LLC file segments.
    
    Args:
        pose_csv_path (str): Path to the pose CSV file
        llc_path (str): Path to the LLC file containing segments
        output_csv_path (str): Path to output labeled CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load segments from LLC file
        segments = load_segments_from_llc(llc_path)
        
        # Load pose CSV
        df = pd.read_csv(pose_csv_path)
        
        # Add stroke_label column (initialized to 0)
        df['stroke_label'] = 0
        
        # Label frames based on segments
        for i, (start_frame, end_frame) in enumerate(segments):
            label = i + 1  # Start labels from 1
            df.loc[(df['frame_index'] >= start_frame) & (df['frame_index'] <= end_frame), 'stroke_label'] = label
        
        # Save labeled CSV
        df.to_csv(output_csv_path, index=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create frame labels: {str(e)}")
        traceback.print_exc()
        return False

def load_labels(csv_path):
    """
    Load frame labels from a CSV file.
    
    Args:
        csv_path (str): Path to the labeled CSV file
        
    Returns:
        list: List of (frame_index, stroke_label) tuples
    """
    try:
        df = pd.read_csv(csv_path)
        labels = list(zip(df['frame_index'], df['stroke_label']))
        return labels
    except Exception as e:
        print(f"[ERROR] Failed to load labels: {str(e)}")
        return []

def get_stroke_segments(labels):
    """
    Extract stroke segments from frame labels.
    
    Args:
        labels (list): List of (frame_index, stroke_label) tuples
        
    Returns:
        list: List of (start_frame, end_frame) tuples for each stroke segment
    """
    segments = []
    current_label = 0
    start_frame = None
    
    for frame, label in labels:
        if label != current_label:
            if current_label > 0 and start_frame is not None:
                # End of a stroke segment
                segments.append((start_frame, frame - 1))
            
            if label > 0:
                # Start of a new stroke segment
                start_frame = frame
            else:
                start_frame = None
                
            current_label = label
    
    # Handle the last segment if it extends to the end
    if current_label > 0 and start_frame is not None:
        segments.append((start_frame, labels[-1][0]))
    
    return segments

def enhanced_clip_video_segments(video_dir, video_name, segments, fps, output_folder):
    """
    Clip all three video types (main, skeleton, and overlay) using the same segmentation data.
    
    Args:
        video_dir (str): Directory containing the videos
        video_name (str): Base name of the video without extension
        segments (list): List of (start_frame, end_frame) tuples
        fps (float): Frames per second of the video
        output_folder (str): Folder to save the clip videos
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Define video paths for all three types
        main_video_path = os.path.join(video_dir, f"{video_name}.mp4")
        skeleton_video_path = os.path.join(video_dir, f"{video_name}_skeleton.mp4")
        overlay_video_path = os.path.join(video_dir, f"{video_name}_overlay.mp4")
        
        # Check which videos exist
        main_exists = os.path.exists(main_video_path)
        skeleton_exists = os.path.exists(skeleton_video_path)
        overlay_exists = os.path.exists(overlay_video_path)
        
        if not main_exists:
            print(f"[ERROR] Main video not found: {main_video_path}")
            return False
            
        print(f"[INFO] Found videos to clip:")
        if main_exists:
            print(f"  - Main video: {os.path.basename(main_video_path)}")
        if skeleton_exists:
            print(f"  - Skeleton video: {os.path.basename(skeleton_video_path)}")
        if overlay_exists:
            print(f"  - Overlay video: {os.path.basename(overlay_video_path)}")
        
        # Process each segment
        for i, (start_frame, end_frame) in enumerate(segments):
            stroke_num = i + 1
            
            # Define output paths for each video type
            main_clip_path = os.path.join(output_folder, f"stroke_{stroke_num}.mp4")
            skeleton_clip_path = os.path.join(output_folder, f"stroke_{stroke_num}_skeleton.mp4")
            overlay_clip_path = os.path.join(output_folder, f"stroke_{stroke_num}_overlay.mp4")
            
            # Calculate timestamps
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            # Clip main video
            if main_exists and not os.path.exists(main_clip_path):
                cmd = [
                    "ffmpeg", "-y", "-i", main_video_path, 
                    "-ss", str(start_time), "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    main_clip_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[INFO] Created main clip {stroke_num}: {os.path.basename(main_clip_path)}")
            elif main_exists:
                print(f"[INFO] Main clip {stroke_num} already exists: {os.path.basename(main_clip_path)}")
            
            # Clip skeleton video
            if skeleton_exists and not os.path.exists(skeleton_clip_path):
                cmd = [
                    "ffmpeg", "-y", "-i", skeleton_video_path, 
                    "-ss", str(start_time), "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    skeleton_clip_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[INFO] Created skeleton clip {stroke_num}: {os.path.basename(skeleton_clip_path)}")
            elif skeleton_exists:
                print(f"[INFO] Skeleton clip {stroke_num} already exists: {os.path.basename(skeleton_clip_path)}")
            
            # Clip overlay video
            if overlay_exists and not os.path.exists(overlay_clip_path):
                cmd = [
                    "ffmpeg", "-y", "-i", overlay_video_path, 
                    "-ss", str(start_time), "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    overlay_clip_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[INFO] Created overlay clip {stroke_num}: {os.path.basename(overlay_clip_path)}")
            elif overlay_exists:
                print(f"[INFO] Overlay clip {stroke_num} already exists: {os.path.basename(overlay_clip_path)}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to clip video segments: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_clip_csv(csv_path, segments, output_folder):
    """
    Extract CSV data for each clip and save to individual files.
    
    Args:
        csv_path (str): Path to the labeled CSV file
        segments (list): List of (start_frame, end_frame) tuples
        output_folder (str): Folder to save the clip CSVs
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        for i, (start_frame, end_frame) in enumerate(segments):
            clip_csv_path = os.path.join(output_folder, f"stroke_{i+1}.csv")
            
            # Skip if CSV already exists
            if os.path.exists(clip_csv_path):
                print(f"[INFO] Clip CSV already exists: {clip_csv_path}")
                continue
            
            # Extract frames for this segment
            clip_df = df[(df['frame_index'] >= start_frame) & (df['frame_index'] <= end_frame)].copy()
            
            # Reset frame index to start from 0
            clip_df['frame_index'] = clip_df['frame_index'] - start_frame
            
            # Save clip CSV
            clip_df.to_csv(clip_csv_path, index=False)
            
            print(f"[INFO] Created clip CSV {i+1}: {clip_csv_path}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract clip CSVs: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def time_normalize_csv(csv_path, output_path, num_frames=RESAMPLED_FRAMES):
    """
    Time-normalize a clip CSV by resampling to a fixed number of frames.
    
    Args:
        csv_path (str): Path to the clip CSV
        output_path (str): Path to save the normalized CSV
        num_frames (int): Number of frames to resample to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Skip if already normalized
        if len(df) == num_frames:
            shutil.copy(csv_path, output_path)
            return True
            
        # Get original frame indices
        orig_frames = df['frame_index'].values
        
        # Create new frame indices (evenly spaced)
        new_frames = np.linspace(orig_frames.min(), orig_frames.max(), num_frames)
        
        # Create new dataframe
        new_df = pd.DataFrame()
        new_df['frame_index'] = np.arange(num_frames)
        
        # Interpolate each column except frame_index and stroke_label
        exclude_cols = ['frame_index', 'stroke_label']
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            # Create interpolation function
            interp_func = interp1d(
                orig_frames, 
                df[col].values, 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            # Apply interpolation
            new_df[col] = interp_func(new_frames)
        
        # Copy stroke_label (if it exists)
        if 'stroke_label' in df.columns:
            # Get most common label
            most_common_label = df['stroke_label'].mode().iloc[0]
            new_df['stroke_label'] = most_common_label
        
        # Save normalized CSV
        new_df.to_csv(output_path, index=False)
        
        print(f"[INFO] Created normalized CSV: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to normalize CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_video_folder_for_segmentation(folder_path):
    """
    Process all videos in a folder for stroke segmentation.
    
    Args:
        folder_path (str): Path to the folder containing videos
        
    Returns:
        bool: True if at least one video was processed successfully
    """
    try:
        # Check if folder exists
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"[ERROR] Folder does not exist: {folder_path}")
            return False
        
        # Get all video files
        video_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov')) 
                      and not f.endswith('_overlay.mp4') 
                      and not f.endswith('_skeleton.mp4')]
        
        if not video_files:
            print(f"[INFO] No video files found in {folder_path}")
            return False
        
        print(f"[INFO] Found {len(video_files)} video files in {folder_path}")
        
        success_count = 0
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            print(f"[INFO] Processing video: {video_file}")
            
            # Skip videos that have been fully processed
            video_name = os.path.splitext(video_file)[0]
            status_path = os.path.join(folder_path, "status.txt")
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    status_content = f.read()
                    if "is_fully_processed: True" in status_content:
                        print(f"[INFO] Skipping {video_file} - already fully processed")
                        success_count += 1
                        continue
            
            # Run segmentation on this video
            if run_stroke_segmentation(video_path, folder_mode=False):
                success_count += 1
            else:
                print(f"[WARN] Failed to process {video_file}")
        
        print(f"[DONE] Successfully processed {success_count}/{len(video_files)} videos")
        return success_count > 0
        
    except Exception as e:
        print(f"[ERROR] Failed to process video folder: {str(e)}")
        traceback.print_exc()
        return False

def run_stroke_segmentation(video_path=None, folder_mode=False):
    """
    Run stroke segmentation on a video to create individual stroke clips.
    
    In LosslessCut, the "cutSegments" define parts to cut out (non-strokes).
    This function extracts the parts BETWEEN those cut segments, which are
    the actual tennis strokes we want to keep and analyze.
    
    Args:
        video_path (str): Path to the video file
        folder_mode (bool): Whether to process all videos in a folder
                            
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Process a single video
        if not video_path:
            print("[ERROR] No video path provided")
            return False
        
        # Extract video information
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Define paths
        llc_path = os.path.join(video_dir, f"{video_name}.llc")
        clips_folder = os.path.join(video_dir, f"{video_name}_clips")
        
        # Create folders if they don't exist
        os.makedirs(clips_folder, exist_ok=True)
        
        print(f"[INFO] Running segmentation on {video_path}")
        print(f"[INFO] Clips will be saved to {clips_folder}")
        
        # Check if necessary files exist
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return False
        
        if not os.path.exists(llc_path):
            print(f"[ERROR] LLC file not found: {llc_path}")
            return False
        
        # Step 1: Load segments directly from LLC file
        segments = load_segments_from_llc(llc_path)
        if not segments:
            print(f"[ERROR] No segments found in LLC file: {llc_path}")
            return False
        
        print(f"[INFO] Found {len(segments)} segments in LLC file")
        
        # Step 2: Create video clips for each segment (all video types)
        # Get video fps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        cap.release()
        
        # Find all available video files
        main_video = video_path
        skeleton_video = os.path.join(video_dir, f"{video_name}_skeleton.mp4")
        overlay_video = os.path.join(video_dir, f"{video_name}_overlay.mp4")
        
        # Track which video types exist
        has_skeleton = os.path.exists(skeleton_video)
        has_overlay = os.path.exists(overlay_video)
        
        print(f"[INFO] Processing {len(segments)} segments from main video: {os.path.basename(main_video)}")
        if has_skeleton:
            print(f"[INFO] Skeleton video found: {os.path.basename(skeleton_video)}")
        if has_overlay:
            print(f"[INFO] Overlay video found: {os.path.basename(overlay_video)}")
        
        # Process each segment
        for i, (start_frame, end_frame) in enumerate(segments):
            segment_num = i + 1
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            print(f"[INFO] Processing segment {segment_num}: {start_time:.2f}s to {end_time:.2f}s")
            
            # Define output files
            main_output = os.path.join(clips_folder, f"stroke_{segment_num}.mp4")
            skeleton_output = os.path.join(clips_folder, f"stroke_{segment_num}_skeleton.mp4")
            overlay_output = os.path.join(clips_folder, f"stroke_{segment_num}_overlay.mp4")
            
            # Skip existing files unless forced to reprocess
            if not os.path.exists(main_output):
                try:
                    # Clip main video
                    cmd = [
                        "ffmpeg", "-y", "-i", main_video, 
                        "-ss", str(start_time), "-t", str(duration),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        main_output
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[INFO] Created main clip: {os.path.basename(main_output)}")
                except Exception as e:
                    print(f"[ERROR] Failed to clip main video: {e}")
            
            # Clip skeleton video if it exists
            if has_skeleton and not os.path.exists(skeleton_output):
                try:
                    cmd = [
                        "ffmpeg", "-y", "-i", skeleton_video, 
                        "-ss", str(start_time), "-t", str(duration),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        skeleton_output
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[INFO] Created skeleton clip: {os.path.basename(skeleton_output)}")
                except Exception as e:
                    print(f"[ERROR] Failed to clip skeleton video: {e}")
            
            # Clip overlay video if it exists
            if has_overlay and not os.path.exists(overlay_output):
                try:
                    cmd = [
                        "ffmpeg", "-y", "-i", overlay_video, 
                        "-ss", str(start_time), "-t", str(duration),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        overlay_output
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[INFO] Created overlay clip: {os.path.basename(overlay_output)}")
                except Exception as e:
                    print(f"[ERROR] Failed to clip overlay video: {e}")
        
        # Step 3: Update the status file
        status_path = os.path.join(video_dir, "status.txt")
        try:
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    status_data = f.read()
                
                # Update the status
                if "is_fully_processed:" in status_data:
                    status_data = re.sub(r"is_fully_processed:.*", f"is_fully_processed: True", status_data)
                else:
                    status_data += "\nis_fully_processed: True"
                
                # Update number of strokes
                if "num_strokes:" in status_data:
                    status_data = re.sub(r"num_strokes:.*", f"num_strokes: {len(segments)}", status_data)
                else:
                    status_data += f"\nnum_strokes: {len(segments)}"
                
                with open(status_path, 'w') as f:
                    f.write(status_data)
            else:
                with open(status_path, 'w') as f:
                    f.write(f"is_fully_processed: True\nnum_strokes: {len(segments)}")
        except Exception as e:
            print(f"[WARN] Failed to update status file: {e}")
        
        print(f"[DONE] Processed {video_name} - created {len(segments)} clips")
        return True
        
    except Exception as e:
        print(f"[ERROR] Stroke segmentation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def manually_copy_clips_to_library(video_id):
    """
    Copy clips from video_X_clips folder to the Strokes Library.
    
    Args:
        video_id: Video ID
        
    Returns:
        List of strokes that were created
    """
    print(f"[INFO] Copying clips to Strokes Library for video_{video_id}")
    
    # Define base paths
    video_dir = os.path.join(VIDEOS_DIR, f"video_{video_id}")
    clips_folder = os.path.join(video_dir, f"video_{video_id}_clips")
    
    if not os.path.exists(clips_folder):
        print(f"[ERROR] Clips folder not found: {clips_folder}")
        return []
    
    print(f"[INFO] Looking for clips in: {clips_folder}")
    
    # Find normalized CSV from the video directory
    video_norm_csv = os.path.join(video_dir, f"video_{video_id}_normalized.csv")
    if not os.path.exists(video_norm_csv):
        print(f"[WARNING] Normalized CSV not found: {video_norm_csv}")
    else:
        print(f"[INFO] Found normalized CSV data")
    
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
        except (ValueError, IndexError) as e:
            print(f"[WARN] Error finding next stroke ID: {e}, using default: {next_stroke_id}")
    
    print(f"[INFO] Next available stroke ID: {next_stroke_id}")
    
    # Find all clip files
    try:
        clip_files = os.listdir(clips_folder)
    except Exception as e:
        print(f"[ERROR] Failed to list clips folder contents: {e}")
        return []
        
    if not clip_files:
        print(f"[ERROR] No clip files found in {clips_folder}")
        return []
        
    # Group files by stroke number
    strokes = {}
    
    # Check regular MP4 files
    mp4_files = [f for f in clip_files if f.endswith('.mp4') and not (f.endswith('_skeleton.mp4') or f.endswith('_overlay.mp4'))]
    print(f"[INFO] Found {len(mp4_files)} regular MP4 files")
    
    # Look for skeleton and overlay clips
    skeleton_files = [f for f in clip_files if f.endswith('_skeleton.mp4')]
    overlay_files = [f for f in clip_files if f.endswith('_overlay.mp4')]
    print(f"[INFO] Found {len(skeleton_files)} skeleton files and {len(overlay_files)} overlay files")
    
    # Group files by stroke number
    for mp4_file in mp4_files:
        match = re.match(r'stroke_(\d+)\.mp4', mp4_file)
        if match:
            stroke_num = match.group(1)
            if stroke_num not in strokes:
                strokes[stroke_num] = {'mp4': None, 'skeleton': None, 'overlay': None}
            strokes[stroke_num]['mp4'] = mp4_file
    
    # Add skeleton files
    for skeleton_file in skeleton_files:
        match = re.match(r'stroke_(\d+)_skeleton\.mp4', skeleton_file)
        if match:
            stroke_num = match.group(1)
            if stroke_num not in strokes:
                strokes[stroke_num] = {'mp4': None, 'skeleton': None, 'overlay': None}
            strokes[stroke_num]['skeleton'] = skeleton_file
    
    # Add overlay files
    for overlay_file in overlay_files:
        match = re.match(r'stroke_(\d+)_overlay\.mp4', overlay_file)
        if match:
            stroke_num = match.group(1)
            if stroke_num not in strokes:
                strokes[stroke_num] = {'mp4': None, 'skeleton': None, 'overlay': None}
            strokes[stroke_num]['overlay'] = overlay_file
    
    print(f"[INFO] Found {len(strokes)} strokes to process")
    
    created_strokes = []
    
    # Copy each stroke to the Strokes Library
    for stroke_num, files in strokes.items():
        stroke_id = next_stroke_id
        next_stroke_id += 1
        
        stroke_folder = os.path.join(STROKES_LIBRARY, f"stroke_{stroke_id}")
        try:
            os.makedirs(stroke_folder, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Failed to create stroke folder: {e}")
            continue
        
        print(f"[INFO] Creating stroke_{stroke_id} in Strokes Library")
        
        # Track if we successfully created this stroke
        success = False
        
        # Copy regular MP4 if available
        if files['mp4']:
            src_file = os.path.join(clips_folder, files['mp4'])
            dest_file = os.path.join(stroke_folder, "stroke_clip.mp4")
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"[INFO] Copied main video clip")
                    success = True
                    
                    # Also create a "raw" version
                    raw_file = os.path.join(stroke_folder, "stroke_raw.mp4")
                    shutil.copy2(src_file, raw_file)
                    print(f"[INFO] Created raw video reference")
                except Exception as e:
                    print(f"[ERROR] Failed to copy main clip: {e}")
        
        # Copy skeleton MP4 if available
        if files['skeleton']:
            src_file = os.path.join(clips_folder, files['skeleton'])
            dest_file = os.path.join(stroke_folder, "stroke_skeleton.mp4")
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"[INFO] Copied skeleton video clip")
                except Exception as e:
                    print(f"[ERROR] Failed to copy skeleton clip: {e}")
        
        # Copy overlay MP4 if available
        if files['overlay']:
            src_file = os.path.join(clips_folder, files['overlay'])
            dest_file = os.path.join(stroke_folder, "stroke_overlay.mp4")
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"[INFO] Copied overlay video clip")
                except Exception as e:
                    print(f"[ERROR] Failed to copy overlay clip: {e}")
            
        # Copy normalized CSV if available
        if os.path.exists(video_norm_csv):
            dest_file = os.path.join(stroke_folder, "stroke_norm.csv")
            try:
                shutil.copy2(video_norm_csv, dest_file)
                print(f"[INFO] Added normalized pose data")
            except Exception as e:
                print(f"[ERROR] Failed to copy normalized data: {e}")
        
        # Create source info file
        source_info_path = os.path.join(stroke_folder, "source_info.txt")
        try:
            with open(source_info_path, 'w') as f:
                f.write(f"Source Video: video_{video_id}\n")
                f.write(f"Stroke Number in Video: {stroke_num}\n")
                f.write(f"Created by merged_tennis_processor.py\n")
                f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"[ERROR] Failed to create source info file: {e}")
        
        if success:
            created_strokes.append(f"stroke_{stroke_id}")
    
    print(f"[DONE] Created {len(created_strokes)} strokes in the Strokes Library")
    return created_strokes

def process_single_video(video_path, llc_path=None, video_id=None, cleanup=True):
    """
    Process a single video with enhanced pose detection, visualization, and segmentation
    
    Args:
        video_path: Path to the input video
        llc_path: Path to the LLC file
        video_id: Video ID to use
        cleanup: Whether to clean up intermediate files (default: True)
        
    Returns:
        bool: Success or failure
    """
    try:
        # Get video ID if not provided
        if video_id is None:
            video_id = get_next_video_id()
        
        videos_dir_path = os.path.join(VIDEOS_DIR, f"video_{video_id}")
        ensure_directory_exists(videos_dir_path)
        
        # Target paths in videos directory
        target_video = os.path.join(videos_dir_path, f"video_{video_id}.mp4")
        target_llc = os.path.join(videos_dir_path, f"video_{video_id}.llc")
        
        print(f"\n=== PROCESSING VIDEO AS video_{video_id} ===")
        
        # Copy or move original video and LLC file to videos directory
        if os.path.abspath(video_path) != os.path.abspath(target_video):
            try:
                shutil.copy2(video_path, target_video)
                print(f"[INFO] Copied original video to {target_video}")
            except Exception as e:
                print(f"[ERROR] Failed to copy video: {e}")
                return False
        
        if llc_path and os.path.exists(llc_path) and os.path.abspath(llc_path) != os.path.abspath(target_llc):
            try:
                shutil.copy2(llc_path, target_llc)
                print(f"[INFO] Copied LLC file to {target_llc}")
            except Exception as e:
                print(f"[ERROR] Failed to copy LLC file: {e}")
                return False
        
        # Step 1: Process video with MediaPipe to create enhanced visualizations
        # Output directly to videos directory
        print(f"[STEP 1] Processing video with MediaPipe")
        skeleton_output, overlay_output, landmarks_dict_list, csv_output = process_video_with_mediapipe(
            target_video, video_id=video_id)
        
        if not csv_output or not os.path.exists(csv_output):
            print(f"[ERROR] Failed to process video or create CSV output")
            return False
        
        # Step 2: Normalize data directly to videos directory
        print(f"[STEP 2] Normalizing pose data")
        norm_csv = normalize_pose_data(csv_output, video_id=video_id)
        
        if not norm_csv or not os.path.exists(norm_csv):
            print(f"[ERROR] Failed to normalize data")
            return False
        
        # Create/update the status file with all information
        status_file = os.path.join(videos_dir_path, "status.txt")
        try:
            with open(status_file, 'w') as f:
                f.write(f"original_filename: {os.path.basename(video_path)}\n")
                f.write(f"processed_date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"has_normalized_data: True\n")
                f.write(f"has_skeleton: {os.path.exists(skeleton_output)}\n")
                f.write(f"has_overlay: {os.path.exists(overlay_output)}\n")
                f.write(f"has_llc: {os.path.exists(target_llc)}\n")
                f.write("is_ready_for_clipping: True\n")
                f.write("is_fully_processed: False\n")
        except Exception as e:
            print(f"[WARN] Failed to create status file: {e}")
        
        # If LLC path is provided, run segmentation
        if llc_path and os.path.exists(target_llc):
            # Check if LLC file has valid annotations
            if check_llc_file(target_llc):
                print(f"[STEP 3] Running stroke segmentation")
                
                # Run segmentation with the correct video path
                if os.path.exists(target_video):
                    success = run_stroke_segmentation(target_video)
                    
                    if success:
                        # Create strokes in library
                        print(f"[STEP 4] Copying strokes to library")
                        created_strokes = manually_copy_clips_to_library(video_id)
                        
                        if created_strokes:
                            print(f"[INFO] Successfully created {len(created_strokes)} strokes in the library.")
                            
                            # Update status file to mark as fully processed
                            try:
                                with open(status_file, 'r') as f:
                                    status_data = f.read()
                                
                                status_data = re.sub(r"is_fully_processed:.*", "is_fully_processed: True", status_data)
                                
                                with open(status_file, 'w') as f:
                                    f.write(status_data)
                            except Exception as e:
                                print(f"[WARN] Failed to update status file: {e}")
                            
                            # Clean up intermediate files if requested
                            if cleanup:
                                print(f"[STEP 5] Cleaning up intermediate files")
                                cleanup_intermediate_files(video_id)
                        else:
                            print(f"[WARN] No strokes were created in the library.")
                    else:
                        print(f"[ERROR] Segmentation failed for video_{video_id}")
                else:
                    print(f"[ERROR] Video file not found: {target_video}")
            else:
                print(f"[WARN] LLC file has no valid stroke annotations, skipping segmentation")
        else:
            print(f"[WARN] No LLC file provided, skipping segmentation")
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Processed {os.path.basename(video_path)} as video_{video_id}")
        print(f"Results available in Strokes_Library directory")
        
        return True
    except Exception as e:
        import traceback
        print(f"[ERROR] Processing video: {e}")
        print(traceback.format_exc())
        return False

def run_segmentation_on_video_id(video_id):
    """
    Run stroke segmentation on a specific video ID.
    
    Args:
        video_id (int): The video ID to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine paths
    video_dir = os.path.join(VIDEOS_DIR, f"video_{video_id}")
    if not os.path.exists(video_dir):
        print(f"[ERROR] Video directory not found: {video_dir}")
        return False
    
    video_path = os.path.join(video_dir, f"video_{video_id}.mp4")
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False
    
    print(f"[INFO] Running segmentation on video_{video_id}")
    
    # Run the segmentation on the video
    success = run_stroke_segmentation(video_path)
    
    if success:
        print(f"[INFO] Segmentation complete for video_{video_id}")
        # Manually copy clips to the library
        strokes = manually_copy_clips_to_library(video_id)
        print(f"[INFO] Created {len(strokes)} strokes in the library")
        return True
    else:
        print(f"[ERROR] Segmentation failed for video_{video_id}")
        return False

def main():
    """Main function that processes all video/LLC pairs in the unprocessed videos folder."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tennis Video Processor')
    parser.add_argument('--input_path', type=str, help='Path to input video')
    parser.add_argument('--llc_path', type=str, help='Path to LLC file (if different from default)')
    parser.add_argument('--video-id', type=int, help='Specify a starting video ID instead of auto-detecting')
    parser.add_argument('--single', action='store_true', help='Process only the first video found')
    parser.add_argument('--segment-video-id', type=int, help='Run segmentation on a specific video ID')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not clean up intermediate files after processing')
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    ensure_directory_exists(UNPROCESSED_DIR)
    ensure_directory_exists(VIDEOS_DIR)
    ensure_directory_exists(STROKES_LIBRARY)
    ensure_directory_exists(OUTPUT_DIR)
    
    # If segment-video-id is provided, run segmentation on that specific video
    if args.segment_video_id is not None:
        video_path = os.path.join(VIDEOS_DIR, f"video_{args.segment_video_id}", f"video_{args.segment_video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return 1
        
        success = run_stroke_segmentation(video_path)
        if success:
            created_strokes = manually_copy_clips_to_library(args.segment_video_id)
            if created_strokes:
                print(f"[INFO] Created {len(created_strokes)} strokes")
            if not args.no_cleanup:
                cleanup_intermediate_files(args.segment_video_id)
        return 0 if success else 1
    
    # If input path is provided, process just that video
    if args.input_path:
        if not os.path.exists(args.input_path):
            print(f"[ERROR] Input video not found: {args.input_path}")
            return 1
            
        # If LLC path is not provided, look for one with same name
        llc_path = args.llc_path
        if not llc_path:
            base_path = os.path.splitext(args.input_path)[0]
            potential_llc = f"{base_path}.llc"
            if os.path.exists(potential_llc):
                llc_path = potential_llc
                print(f"[INFO] Found LLC file: {llc_path}")
            else:
                print(f"[WARN] No LLC file found for {args.input_path}")
        
        # Process the video (cleanup is true by default unless --no-cleanup is specified)
        success = process_single_video(
            args.input_path, 
            llc_path=llc_path, 
            video_id=args.video_id,
            cleanup=not args.no_cleanup
        )
        
        return 0 if success else 1
    
    # If no input path is provided, look for video+LLC pairs in the unprocessed directory
    video_llc_pairs = get_video_and_llc_paths()
    
    if not video_llc_pairs:
        print("No video+LLC pairs found in the unprocessed_videos directory.")
        print("Please add video files and their corresponding .llc files.")
        parser.print_help()
        return 1
    
    print(f"Found {len(video_llc_pairs)} video+LLC pairs to process.")
    
    # Process videos
    video_id = args.video_id if args.video_id is not None else get_next_video_id()
    processed_count = 0
    
    for video_path, llc_path in video_llc_pairs:
        current_video_id = video_id + processed_count
        print(f"\n[INFO] Processing pair {processed_count+1}/{len(video_llc_pairs)}: {os.path.basename(video_path)}")
        
        # Process the video (cleanup is true by default unless --no-cleanup is specified)
        success = process_single_video(
            video_path, 
            llc_path=llc_path, 
            video_id=current_video_id,
            cleanup=not args.no_cleanup
        )
        
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
                print(f"[INFO] Moved processed files to archive folder")
            except Exception as e:
                print(f"[WARN] Could not move processed files to archive: {e}")
        
        if args.single:
            break  # Only process the first video if --single flag is set
    
    print(f"\n=== SUMMARY ===")
    print(f"Total videos processed: {processed_count}")
    print(f"Check the {STROKES_LIBRARY} directory for results.")
    
    return 0

def cleanup_intermediate_files(video_id):
    """
    Clean up intermediate files and folders after processing to save disk space.
    
    This function removes:
    1. Temporary files in the Output directory
    2. Clips folder in the videos/video_X directory (after copying to Strokes_Library)
    
    Args:
        video_id: Video ID to clean up
        
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    try:
        print(f"[INFO] Cleaning up intermediate files for video_{video_id}")
        
        # Clean up Output directory
        output_prefix = f"video_{video_id}"
        output_files = [
            f"{OUTPUT_DIR}/{output_prefix}_skeleton.mp4",
            f"{OUTPUT_DIR}/{output_prefix}_overlay.mp4",
            f"{OUTPUT_DIR}/{output_prefix}_data.csv",
            f"{OUTPUT_DIR}/{output_prefix}_normalized.csv"
        ]
        
        output_removed = 0
        for file_path in output_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    output_removed += 1
                    print(f"[INFO] Removed {file_path}")
                except Exception as e:
                    print(f"[WARN] Failed to remove {file_path}: {e}")
        
        # Remove clips folder if it exists in videos directory
        video_dir = os.path.join(VIDEOS_DIR, f"video_{video_id}")
        clips_folder = os.path.join(video_dir, f"video_{video_id}_clips")
        
        clips_removed = False
        if os.path.exists(clips_folder):
            try:
                shutil.rmtree(clips_folder)
                clips_removed = True
                print(f"[INFO] Removed clips folder: {clips_folder}")
            except Exception as e:
                print(f"[WARN] Failed to remove clips folder: {e}")
        
        print(f"[DONE] Cleaned up intermediate files for video_{video_id}")
        print(f"       - Removed {output_removed} temporary files from Output directory")
        print(f"       - Clips folder removed: {clips_removed}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(main()) 