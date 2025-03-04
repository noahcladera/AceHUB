#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3Full_Video_Normalization.py

This script processes each video folder in the "data" directory.
For each subfolder (e.g., "video_1") containing a raw pose CSV file (named like "video_1_data.csv"),
it applies spatial normalization and temporal smoothing to add additional columns:
  norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase.
The output is saved as "video_1_normalized.csv" in the same folder.
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
# Base folder containing video subfolders (each video has its own folder)
BASE_DATA_DIR = "data"
# Temporal processing settings
TEMPORAL_WINDOW = 30   # Number of frames for sliding window
FPS = 30               # Frames per second of the video
# Normalization settings
VISIBILITY_THRESHOLD = 0.5  # Only consider landmarks with visibility > threshold

#############################################
# SPATIAL NORMALIZATION FUNCTIONS
#############################################
def normalize_pose(row):
    """
    Converts a CSV row (raw pose data) into normalized pose data.
    Expects the row to contain: frame_index followed by 33*4 values for the landmarks.
    Returns a dict with normalized landmarks and additional info.
    """
    # Build list of 33 landmarks as numpy arrays
    landmarks = []
    for i in range(33):
        base = 1 + i * 4
        lm = np.array([
            float(row[base]),     # x
            float(row[base+1]),   # y
            float(row[base+2]),   # z
            float(row[base+3])    # visibility
        ])
        landmarks.append(lm)
    
    # Compute hip center from landmarks 23 and 24 (first 3 dims)
    hip_center = (landmarks[23][:3] + landmarks[24][:3]) / 2
    # Compute shoulder center from landmarks 11 and 12
    shoulder_center = (landmarks[11][:3] + landmarks[12][:3]) / 2
    torso_vector = shoulder_center - hip_center
    rotation_angle = math.atan2(torso_vector[1], torso_vector[0])
    rot_matrix = Rotation.from_euler('z', -rotation_angle).as_matrix()[0:2, 0:2]
    torso_length = np.linalg.norm(torso_vector)
    
    # Normalize each landmark (for x and y, scale and rotate; z is scaled)
    normalized = {}
    for idx, lm in enumerate(landmarks):
        if lm[3] < VISIBILITY_THRESHOLD:
            normalized[idx] = [np.nan, np.nan, np.nan]
            continue
        centered = lm[:2] - hip_center[:2]
        rotated = rot_matrix @ centered
        scaled = rotated / torso_length
        norm_z = lm[2] / torso_length
        normalized[idx] = [scaled[0], scaled[1], norm_z]
    
    court_position = normalize_to_court(hip_center)
    return {
        'normalized_landmarks': normalized,
        'hip_center': hip_center,
        'torso_length': torso_length,
        'body_rotation': rot_matrix.tolist(),
        'court_position': court_position
    }

def normalize_to_court(hip_center):
    """
    Dummy court normalization based on hip center.
    """
    return {
        'court_x': hip_center[0] * 1.5,
        'court_y': hip_center[1] * 0.8
    }

#############################################
# TEMPORAL NORMALIZATION FUNCTIONS
#############################################
def calculate_derivative(data, fps):
    return np.gradient(data, axis=0) * fps

def smooth_landmark_trajectories(frames, window=15, order=3):
    from scipy.signal import savgol_filter
    return savgol_filter(frames, window_length=window, polyorder=order, axis=0)

def process_temporal_features(frames, fps=30):
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
def detect_swing_phase(normalized, temporal):
    """
    Uses vertical velocity of landmark 16 (right wrist) to roughly determine stroke phase.
    Returns 'backswing', 'forward_swing', or 'neutral'.
    """
    try:
        vel_last = temporal['velocity'][-1]
        idx = 16 * 3 + 1  # For landmark 16, y is at this index in the flattened list.
        vy = vel_last[idx]
        if vy > 0.5:
            return 'backswing'
        elif vy < -0.5:
            return 'forward_swing'
    except Exception as e:
        print("Error in swing phase detection:", e)
    return 'neutral'

#############################################
# FULL INTEGRATION: PROCESS CSV FILES
#############################################
def flatten_normalized_landmarks(normalized_landmarks):
    """
    Flattens the normalized landmarks dict into a single list.
    """
    flat = []
    for i in range(33):
        vals = normalized_landmarks.get(i, [np.nan, np.nan, np.nan])
        flat.extend(vals)
    return flat

def process_csv(input_csv, output_csv, fps=30):
    """
    Reads a raw pose CSV file, applies normalization, and writes a new CSV
    with additional columns: norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase.
    """
    with open(input_csv, 'r') as infile:
        reader = csv.reader(infile)
        original_header = next(reader)
        raw_rows = list(reader)
    
    output_data = []
    frame_buffer = []  # For accumulating frames for temporal processing
    temporal_features = None

    for row in raw_rows:
        norm_info = normalize_pose(row)
        frame_buffer.append(flatten_normalized_landmarks(norm_info['normalized_landmarks']))
        if len(frame_buffer) >= TEMPORAL_WINDOW:
            temporal_features = process_temporal_features(frame_buffer, fps=fps)
            frame_buffer.pop(0)
        swing_phase = detect_swing_phase(norm_info, temporal_features) if temporal_features is not None else 'neutral'
        hip_center = norm_info['hip_center']
        court_pos = norm_info['court_position']
        torso_length = norm_info['torso_length']
        extra = [hip_center[0], hip_center[1], court_pos['court_x'], court_pos['court_y'], torso_length, swing_phase]
        output_data.append(row + [str(e) for e in extra])
    
    new_header = original_header + ['norm_hip_x', 'norm_hip_y', 'court_x', 'court_y', 'torso_length', 'swing_phase']
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(output_data)
    print(f"Normalized data saved to {output_csv}")

def process_all_csv_files(base_folder, fps=30):
    """
    Iterates over each video folder in the base_folder.
    For each folder, finds the raw pose CSV (ending with '_data.csv') and creates a normalized CSV
    (replacing '_data.csv' with '_normalized.csv').
    """
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        data_csv = None
        for f in os.listdir(folder_path):
            if f.endswith("_data.csv"):
                data_csv = os.path.join(folder_path, f)
                break
        if data_csv is None:
            print(f"[WARNING] No _data.csv found in {folder_path}. Skipping.")
            continue
        output_csv = data_csv.replace("_data.csv", "_normalized.csv")
        process_csv(data_csv, output_csv, fps=fps)

#############################################
# MAIN EXECUTION
#############################################
if __name__ == '__main__':
    process_all_csv_files(BASE_DATA_DIR, fps=FPS)
