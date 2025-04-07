#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
normalization_single_hardcoded_height.py

Processes one CSV file with raw pose data (e.g. "my_data.csv"),
applies rotation + normalization based on the distance from top of head (lm_0)
to foot (lm_32), plus optional temporal smoothing. Writes an output CSV
(e.g. "my_data_normalized.csv").

This version replaces torso_length-based scaling with a "full height" approach.

No scanning of subfolders or command-line args – 
just edit the paths below.

Dependencies:
  pip install numpy scipy dtaidistance
"""

import os
import csv
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
# from dtaidistance import dtw  # If used for advanced time alignment
import cv2  # If needed, else remove

#############################################
# USER SETTINGS: Hard-coded paths
#############################################
INPUT_CSV = "data/personal/Video Trimmer Cut Video_data.csv"
OUTPUT_CSV = "my_clip_data_normalized.csv"

FPS = 30               # Frames per second
TEMPORAL_WINDOW = 30   # Number of frames for sliding window in temporal analysis
VISIBILITY_THRESHOLD = 0.5

#############################################
# NORMALIZATION FUNCTIONS
#############################################

def normalize_pose(row):
    """
    Converts a single CSV row (frame_index + 33 landmarks × 4 columns)
    into a dictionary with 'normalized_landmarks', etc.

    Steps:
      1) Identify hip_center from landmarks 23,24
      2) SHIFT => so hip_center is origin
      3) Rotate => align the torso horizontally
      4) Compute full height => from landmark 0 (nose) to 32 (right foot)
      5) If that fails (invisible or zero-len), fallback to torso_length
      6) Scale => normalized coords
    """
    # Build a list of 33 landmarks as np arrays: [x,y,z,visibility]
    landmarks = []
    for i in range(33):
        base = 1 + i*4  # skip frame_index
        lm = np.array([
            float(row[base]),
            float(row[base + 1]),
            float(row[base + 2]),
            float(row[base + 3])
        ])
        landmarks.append(lm)

    # 1) Compute hip_center (landmarks 23 & 24)
    hip_center = (landmarks[23][:3] + landmarks[24][:3]) / 2.0

    # 2) Shoulder center (11 & 12) => define a "torso_vector" for rotation
    shoulder_center = (landmarks[11][:3] + landmarks[12][:3]) / 2.0
    torso_vector = shoulder_center - hip_center

    # Rotation angle => align the torso horizontally
    rotation_angle = math.atan2(torso_vector[1], torso_vector[0])  # y,x
    rot_matrix_2x2 = Rotation.from_euler('z', -rotation_angle).as_matrix()[0:2, 0:2]
    torso_length = np.linalg.norm(torso_vector)

    # We'll store normalized coords here
    normalized = {}

    # 3) Shift & rotate every landmark
    #    We'll do it in two steps: shift => rotate => store in an array for height measurement
    rotated_landmarks_2d = [None]*33
    for idx, lm in enumerate(landmarks):
        if lm[3] < VISIBILITY_THRESHOLD:
            rotated_landmarks_2d[idx] = np.array([np.nan, np.nan])
            continue

        # SHIFT => so hip_center is origin
        xy_centered = lm[:2] - hip_center[:2]
        # ROTATE => around z-axis
        xy_rotated = rot_matrix_2x2 @ xy_centered
        rotated_landmarks_2d[idx] = xy_rotated  # shape (2,)

    # 4) Attempt to measure "full height" from top(0) to foot(32) after rotation
    #    ignoring z for now, or we can handle 3D if you want
    #    We'll do it in 2D for simplicity. If you want full 3D, you'd also rotate Z
    #    and measure the distance in 3D.
    #    We'll also check if 0 or 32 is np.nan => fallback
    def visible_2d(coords):
        return not (np.isnan(coords[0]) or np.isnan(coords[1]))

    height_2d = 0.0
    if visible_2d(rotated_landmarks_2d[0]) and visible_2d(rotated_landmarks_2d[32]):
        # measure 2D distance
        top2d = rotated_landmarks_2d[0]
        foot2d = rotated_landmarks_2d[32]
        height_2d = np.linalg.norm(top2d - foot2d)
    # fallback if 0 => use torso_length
    if height_2d <= 1e-6:
        height_2d = torso_length

    # Now scale => each landmark => normalized coords
    # (We won't do anything with z from top->foot, but you might do 3D if you prefer)
    scale_factor = 1.0
    if height_2d != 0:
        scale_factor = 1.0 / height_2d

    # Build final normalized coords
    for idx, lm in enumerate(landmarks):
        if lm[3] < VISIBILITY_THRESHOLD:
            normalized[idx] = [np.nan, np.nan, np.nan]
            continue

        xy_rotated = rotated_landmarks_2d[idx]
        scaled_x = xy_rotated[0] * scale_factor
        scaled_y = xy_rotated[1] * scale_factor

        # For Z, we can also scale by the same factor
        # We didn't rotate Z in this snippet, but let's do an approximate
        # SHIFT => (lm[2] - hip_center[2])? We'll do a quick approach:
        # If you want to rotate Z, you'd handle a full 3D rotation matrix above.
        z_centered = lm[2] - hip_center[2]
        scaled_z = z_centered * scale_factor

        normalized[idx] = [scaled_x, scaled_y, scaled_z]

    # Example "court" position
    court_pos = normalize_to_court(hip_center)

    return {
        'normalized_landmarks': normalized,
        'hip_center': hip_center,
        'torso_length': torso_length,  # we keep this if we want to see the fallback or reference
        'body_rotation': rot_matrix_2x2.tolist(),
        'court_position': court_pos
    }

def normalize_to_court(hip_center):
    """
    Dummy function for "court" alignment - optional
    """
    return {
        'court_x': hip_center[0] * 1.5,
        'court_y': hip_center[1] * 0.8
    }

#############################################
# TEMPORAL SMOOTHING FUNCTIONS
#############################################

def smooth_landmark_trajectories(frames, window=15, order=3):
    """
    Use a Savitzky-Golay filter to smooth the time series.
    frames: np.array of shape (n_frames, n_features)
    """
    return savgol_filter(frames, window_length=window, polyorder=order, axis=0)

def calculate_derivative(data, fps):
    """Compute derivative (velocity/accel) using np.gradient, scaled by fps."""
    return np.gradient(data, axis=0) * fps

def process_temporal_features(frames, fps=30):
    """
    frames: list of flattened coords => shape (n_frames, 99)
    1) Smooth => velocity => acceleration
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
# EXAMPLE "PHASE DETECTION"
#############################################

def detect_swing_phase(norm_info, temporal):
    """
    Example: use vertical velocity of landmark 16 (right wrist)
    to guess 'backswing', 'forward_swing', or 'neutral'.
    """
    if temporal is None:
        return 'neutral'
    try:
        vel_last = temporal['velocity'][-1]
        # landmark 16 => indices (16*3..16*3+2) => 48..50, y=49
        idx_y = 49
        vy = vel_last[idx_y]
        if vy > 0.5:
            return 'backswing'
        elif vy < -0.5:
            return 'forward_swing'
    except:
        pass
    return 'neutral'

def flatten_normalized_landmarks(normalized_landmarks):
    """
    Flatten from {idx -> [x,y,z]} to length=99 (33*3)
    """
    flat = []
    for i in range(33):
        (nx, ny, nz) = normalized_landmarks.get(i, [np.nan, np.nan, np.nan])
        flat.extend([nx, ny, nz])
    return flat

#############################################
# MAIN NORMALIZATION
#############################################

def normalize_csv(input_csv, output_csv, fps=30, temporal_window=30):
    """
    Reads 'input_csv' => do rotation + "height-based" scaling,
    => optional smoothing => detect swing phase => writes 'output_csv'.
    """
    # load entire CSV
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        original_header = next(reader)
        rows = list(reader)

    frame_buffer = []
    output_data = []

    for row in rows:
        norm_info = normalize_pose(row)
        flat_frame = flatten_normalized_landmarks(norm_info['normalized_landmarks'])
        frame_buffer.append(flat_frame)

        # once we have temporal_window frames, compute velocity, etc.
        if len(frame_buffer) >= temporal_window:
            temporal_feats = process_temporal_features(frame_buffer, fps=fps)
            frame_buffer.pop(0)
        else:
            temporal_feats = None

        phase = detect_swing_phase(norm_info, temporal_feats)

        hip_center = norm_info['hip_center']
        # "court_x/y" is optional
        court_pos = norm_info['court_position']
        torso_len = norm_info['torso_length']

        # extra columns
        extra = [
            hip_center[0], hip_center[1],
            court_pos['court_x'], court_pos['court_y'],
            torso_len, phase
        ]
        output_data.append(row + [str(e) for e in extra])

    # build new header
    new_header = original_header + [
        'norm_hip_x','norm_hip_y',
        'court_x','court_y',
        'torso_length','swing_phase'
    ]

    with open(output_csv, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(new_header)
        writer.writerows(output_data)

    print(f"[INFO] Wrote normalized CSV => {output_csv}")

#############################################
# HARDCODED MAIN
#############################################

def main():
    if not os.path.isfile(INPUT_CSV):
        print(f"[ERROR] Input CSV not found: {INPUT_CSV}")
        return
    print(f"[INFO] Normalizing (height-based) => {INPUT_CSV}")
    normalize_csv(INPUT_CSV, OUTPUT_CSV, fps=FPS, temporal_window=TEMPORAL_WINDOW)

if __name__ == "__main__":
    main()
