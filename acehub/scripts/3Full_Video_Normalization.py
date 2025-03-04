#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3Full_Video_Normalization.py

This script processes each video folder in the "data" directory (e.g. "video_1", "video_2", etc.).
For each subfolder that contains a raw pose CSV (named like "video_1_data.csv"),
it applies spatial normalization and temporal smoothing, then saves the result as
"video_1_normalized.csv" in the same folder.

Columns added:
  norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase

Usage:
    python 3Full_Video_Normalization.py
"""

import os
import re
import csv
import math
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from dtaidistance import dtw
import cv2

#############################################
# CONFIGURATION SETTINGS
#############################################

# We'll determine the path to the 'data' folder based on this script's location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

TEMPORAL_WINDOW = 30   # Number of frames for sliding window
FPS = 30               # Frames per second of the video
VISIBILITY_THRESHOLD = 0.5  # Only consider landmarks with visibility > threshold

#############################################
# SPATIAL NORMALIZATION FUNCTIONS
#############################################

def normalize_pose(row):
    """
    Converts a CSV row (raw pose data) into normalized pose data.
    Expects the row to contain: frame_index followed by 33*4 landmark values.
    Returns a dict with normalized landmarks and additional info.
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

    # Dummy court normalization
    court_pos = normalize_to_court(hip_center)

    return {
        'normalized_landmarks': normalized,
        'hip_center': hip_center,
        'torso_length': torso_length,
        'body_rotation': rot_matrix.tolist(),
        'court_position': court_pos
    }

def normalize_to_court(hip_center):
    """Dummy function to represent court alignment."""
    return {
        'court_x': hip_center[0] * 1.5,
        'court_y': hip_center[1] * 0.8
    }

#############################################
# TEMPORAL NORMALIZATION FUNCTIONS
#############################################

def calculate_derivative(data, fps):
    """Compute derivative (velocity/accel) by np.gradient, scaled by fps."""
    return np.gradient(data, axis=0) * fps

def smooth_landmark_trajectories(frames, window=15, order=3):
    """Savitzky–Golay filter to smooth pose trajectories over time."""
    return savgol_filter(frames, window_length=window, polyorder=order, axis=0)

def process_temporal_features(frames, fps=30):
    """
    Takes a list of flattened frames, smooths them, then calculates velocity and acceleration.
    Returns a dict with keys 'positions', 'velocity', 'acceleration'.
    """
    frames = np.array(frames)
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
    Example: uses vertical velocity of landmark 16 (right wrist) to classify stroke phases.
    """
    if temporal is None:
        return 'neutral'
    try:
        vel_last = temporal['velocity'][-1]
        # landmark 16 => flatten index = 16*3 => x => +1 => y => so y is (16*3)+1
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
# FLATTENING & PROCESSING
#############################################

def flatten_normalized_landmarks(normalized_landmarks):
    """
    Flatten from a dict {idx -> [x, y, z]} to a single list of length 33*3.
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
        # Flatten and store in buffer for temporal smoothing
        flat_frame = flatten_normalized_landmarks(norm_info['normalized_landmarks'])
        frame_buffer.append(flat_frame)

        if len(frame_buffer) >= TEMPORAL_WINDOW:
            temporal_features = process_temporal_features(frame_buffer, fps=fps)
            # Slide the window
            frame_buffer.pop(0)

        # Example "swing phase" detection
        phase = detect_swing_phase(norm_info, temporal_features)
        hip_center = norm_info['hip_center']
        court_pos = norm_info['court_position']
        torso_length = norm_info['torso_length']

        # Extra columns
        extra = [hip_center[0], hip_center[1],
                 court_pos['court_x'], court_pos['court_y'],
                 torso_length, phase]

        output_data.append(row + [str(e) for e in extra])

    new_header = original_header + [
        'norm_hip_x', 'norm_hip_y',
        'court_x', 'court_y',
        'torso_length', 'swing_phase'
    ]

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(output_data)

    print(f"[INFO] Normalized data saved to {output_csv}")

def process_all_csv_files(base_folder, fps=30):
    """
    For each subfolder in base_folder (e.g. data/video_1/),
    find a file ending with "_data.csv", produce "_normalized.csv".
    """
    if not os.path.isdir(base_folder):
        print(f"[ERROR] Data folder not found: {base_folder}")
        return

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
        # Overwrite if exists
        if os.path.exists(output_csv):
            os.remove(output_csv)

        process_csv(data_csv, output_csv, fps=fps)


#############################################
# MAIN EXECUTION
#############################################
if __name__ == "__main__":
    process_all_csv_files(BASE_DATA_DIR, fps=FPS)
