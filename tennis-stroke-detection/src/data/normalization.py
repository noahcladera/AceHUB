#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
normalization.py

Processes each video folder in the "tennis-stroke-detection/data/interim"
directory. For each subfolder containing a raw pose CSV (named like
"video_1_data.csv"), it applies spatial normalization and temporal smoothing,
then saves the result as "video_1_normalized.csv" in the same folder.

Columns added:
  norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase

Usage:
    python normalization.py
"""

import os
import re
import csv
import math
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from dtaidistance import dtw  # Remove if not used
import cv2  # Remove if not used

#############################################
# CONFIGURATION SETTINGS
#############################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "interim")

TEMPORAL_WINDOW = 30   # Number of frames for sliding window
FPS = 30               # Frames per second of the video
VISIBILITY_THRESHOLD = 0.5  # Only consider landmarks with visibility > threshold
FORCE_REPROCESS = False  # Set to True to reprocess videos even if normalized files exist

#############################################
# SPATIAL NORMALIZATION FUNCTIONS
#############################################

def normalize_pose(row):
    """
    Converts a CSV row (raw pose data) into normalized pose data.
    Expects the row to contain: frame_index followed by 33*4 landmark values.
    Returns a dict with normalized landmarks and additional info:
      {
        'normalized_landmarks': dict of idx -> [x, y, z] (normalized),
        'hip_center': np.array,
        'torso_length': float,
        'body_rotation': nested list (2x2),
        'court_position': dict with 'court_x', 'court_y'
      }
    """
    # Build list of 33 landmarks as numpy arrays
    landmarks = []
    for i in range(33):
        base = 1 + i * 4  # skip frame_index (the first column)
        lm = np.array([
            float(row[base]),
            float(row[base + 1]),
            float(row[base + 2]),
            float(row[base + 3])
        ])
        landmarks.append(lm)

    # 1) Hip center from landmarks 23 & 24
    hip_center = (landmarks[23][:3] + landmarks[24][:3]) / 2
    # 2) Shoulder center from landmarks 11 & 12
    shoulder_center = (landmarks[11][:3] + landmarks[12][:3]) / 2
    torso_vector = shoulder_center - hip_center

    # Rotation angle to align torso horizontally
    rotation_angle = math.atan2(torso_vector[1], torso_vector[0])
    rot_matrix = Rotation.from_euler('z', -rotation_angle).as_matrix()[0:2, 0:2]
    torso_length = np.linalg.norm(torso_vector)

    # Normalize each landmark
    normalized = {}
    for idx, lm in enumerate(landmarks):
        if lm[3] < VISIBILITY_THRESHOLD:
            # If visibility is too low, store NaNs
            normalized[idx] = [np.nan, np.nan, np.nan]
            continue

        # Translate so hip_center is origin
        centered = lm[:2] - hip_center[:2]
        # Rotate around z-axis
        rotated = rot_matrix @ centered
        # Scale by torso length
        scaled = rotated / torso_length
        norm_z = lm[2] / torso_length

        normalized[idx] = [scaled[0], scaled[1], norm_z]

    # Example court normalization
    court_pos = normalize_to_court(hip_center)

    return {
        'normalized_landmarks': normalized,
        'hip_center': hip_center,
        'torso_length': torso_length,
        'body_rotation': rot_matrix.tolist(),
        'court_position': court_pos
    }

def normalize_to_court(hip_center):
    """
    Dummy function to represent some form of court alignment or transform.
    Modify or remove as needed.
    """
    return {
        'court_x': hip_center[0] * 1.5,
        'court_y': hip_center[1] * 0.8
    }

#############################################
# TEMPORAL NORMALIZATION FUNCTIONS
#############################################

def calculate_derivative(data, fps):
    """
    Compute a derivative (velocity or acceleration) by np.gradient, scaled by fps.
    data: np.array of shape (frames, features)
    fps:  integer
    """
    return np.gradient(data, axis=0) * fps

def smooth_landmark_trajectories(frames, window=15, order=3):
    """
    Uses a Savitzky-Golay filter to smooth pose trajectories over time.
    frames: np.array of shape (frames, features)
    window: smoothing window size
    order:  polynomial order
    """
    return savgol_filter(frames, window_length=window, polyorder=order, axis=0)

def process_temporal_features(frames, fps=30):
    """
    Takes a list of flattened frames (shape: n_frames x 99 if 33 landmarks x 3 coords).
    1) Smooth them with Savitzky-Golay
    2) Calculate velocity (first derivative)
    3) Calculate acceleration (second derivative)
    Returns a dict with keys: 'positions', 'velocity', 'acceleration'
    """
    frames = np.array(frames)  # shape => (num_frames, features)
    smoothed = smooth_landmark_trajectories(frames)
    velocity = calculate_derivative(smoothed, fps)
    acceleration = calculate_derivative(velocity, fps)
    return {
        'positions': smoothed.tolist(),
        'velocity': velocity.tolist(),
        'acceleration': acceleration.tolist()
    }

#############################################
# SWING PHASE DETECTION (EXAMPLE)
#############################################

def detect_swing_phase(norm_info, temporal):
    """
    Example approach: uses the vertical velocity of landmark 16 (right wrist)
    to classify stroke phases ("backswing/forward_swing/neutral").
    """
    if temporal is None:
        return 'neutral'
    try:
        vel_last = temporal['velocity'][-1]
        # landmark 16 => indices (16*3):(16*3+3) => [48, 49, 50] if you flatten x,y,z
        # y is index = 49 (48+1)
        idx_y = (16 * 3) + 1
        vy = vel_last[idx_y]
        if vy > 0.5:
            return 'backswing'
        elif vy < -0.5:
            return 'forward_swing'
    except Exception as e:
        print("[WARNING] Error in detect_swing_phase:", e)
    return 'neutral'

#############################################
# FLATTENING & CSV PROCESS
#############################################

def flatten_normalized_landmarks(normalized_landmarks):
    """
    Flatten from a dict {idx -> [x, y, z]} to a list of length 33*3 = 99.
    """
    flat = []
    for i in range(33):
        vals = normalized_landmarks.get(i, [np.nan, np.nan, np.nan])
        flat.extend(vals)
    return flat

def process_csv(input_csv, output_csv, fps=30):
    """
    Reads the raw pose CSV, normalizes each frame, and appends columns:
        norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase
    """
    with open(input_csv, 'r') as infile:
        reader = csv.reader(infile)
        original_header = next(reader)
        rows = list(reader)

    output_data = []
    frame_buffer = []
    temporal_features = None

    for row in rows:
        norm_info = normalize_pose(row)
        # Flatten and store in a sliding window
        flat_frame = flatten_normalized_landmarks(norm_info['normalized_landmarks'])
        frame_buffer.append(flat_frame)

        # Once we have TEMPORAL_WINDOW frames in the buffer, compute temporal features
        if len(frame_buffer) >= TEMPORAL_WINDOW:
            temporal_features = process_temporal_features(frame_buffer, fps=fps)
            frame_buffer.pop(0)

        # Simple example: detect "swing phase" from the last frame's velocity
        phase = detect_swing_phase(norm_info, temporal_features)
        hip_center = norm_info['hip_center']
        court_pos = norm_info['court_position']
        torso_length = norm_info['torso_length']

        # Additional columns to write
        extra = [hip_center[0], hip_center[1],
                 court_pos['court_x'], court_pos['court_y'],
                 torso_length, phase]

        output_data.append(row + [str(e) for e in extra])

    # Build a new header
    new_header = original_header + [
        'norm_hip_x', 'norm_hip_y',
        'court_x', 'court_y',
        'torso_length', 'swing_phase'
    ]

    # Write result to CSV
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(output_data)

    print(f"[INFO] Normalized data saved to {output_csv}")

def process_all_csv_files(base_folder, fps=30):
    """
    For each subfolder in base_folder (e.g. data/interim/video_1/), find
    a file ending with "_data.csv" and produce "_normalized.csv".
    """
    if not os.path.isdir(base_folder):
        print(f"[ERROR] Data folder not found: {base_folder}")
        return

    processed_count = 0
    skipped_count = 0
    skipped_folders = []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Find the raw CSV
        data_csv = None
        for fname in os.listdir(folder_path):
            if fname.endswith("_data.csv"):
                data_csv = os.path.join(folder_path, fname)
                break

        if data_csv is None:
            print(f"[WARNING] No '_data.csv' found in {folder_path}. Skipping.")
            continue

        output_csv = data_csv.replace("_data.csv", "_normalized.csv")

        # Skip if already exists (unless FORCE_REPROCESS = True)
        if os.path.exists(output_csv) and not FORCE_REPROCESS:
            print(f"[INFO] Normalized file already exists for {folder_name}. Skipping.")
            skipped_count += 1
            skipped_folders.append(folder_name)
            continue

        print(f"[INFO] Processing {folder_name}...")
        process_csv(data_csv, output_csv, fps=fps)
        processed_count += 1

    print(f"\n[SUMMARY] Processed {processed_count} folders, skipped {skipped_count} folders.")
    if skipped_count > 0:
        print(f"[SUMMARY] Skipped folders: {', '.join(skipped_folders)}")

#############################################
# MAIN EXECUTION
#############################################

def main():
    print(f"[INFO] Starting normalization process...")
    print(f"[INFO] Data directory: {BASE_DATA_DIR}")
    print(f"[INFO] Force reprocess: {FORCE_REPROCESS}")
    process_all_csv_files(BASE_DATA_DIR, fps=FPS)

if __name__ == "__main__":
    main()