import os
import re
import csv
import math
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine
from dtaidistance import dtw
import cv2

#############################################
# CONFIGURATION SETTINGS
#############################################
# Folder containing raw CSV files (generated from video pose extraction)
INPUT_FOLDER = "tennis_clips/videos"    # Contains files like "video_1_data.csv"
# Folder where normalized CSV files will be saved
OUTPUT_FOLDER = "tennis_clips/normalized"   
# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Normalization settings
VISIBILITY_THRESHOLD = 0.5       # Only consider landmarks with visibility > threshold
# (We use shoulders: indices 11 and 12; hips: 23 and 24 as our reference points)

# Temporal processing settings
TEMPORAL_WINDOW = 30             # Number of frames for sliding window (e.g., 30 frames ~ 1 second at 30 FPS)
FPS = 30                         # Frames per second of the video

#############################################
# SPATIAL NORMALIZATION FUNCTIONS
#############################################
def normalize_pose(row):
    """
    Converts a CSV row (raw pose data) into normalized pose data.
    Expects row to have: frame_index, then 33*4 columns for landmarks:
      [lm_0_x, lm_0_y, lm_0_z, lm_0_vis, lm_1_x, ... lm_32_vis].
    Returns a dict with:
       'normalized_landmarks': dict mapping landmark index -> [norm_x, norm_y, norm_z]
       'hip_center': numpy array of hip center (from landmarks 23 & 24)
       'torso_length': scalar torso length
       'body_rotation': rotation matrix (as list) used for alignment
       'court_position': placeholder result based on hip center
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
    
    # 1. Define hip center from landmarks 23 and 24 (first 3 dimensions)
    hip_center = (landmarks[23][:3] + landmarks[24][:3]) / 2
    
    # 2. Define shoulder center from landmarks 11 and 12
    shoulder_center = (landmarks[11][:3] + landmarks[12][:3]) / 2
    torso_vector = shoulder_center - hip_center
    # Compute rotation to align the torso horizontally (along the x-axis)
    rotation_angle = np.arctan2(torso_vector[1], torso_vector[0])
    rot_matrix = Rotation.from_euler('z', -rotation_angle).as_matrix()[0:2, 0:2]
    
    # 3. Use the torso length for scaling
    torso_length = np.linalg.norm(torso_vector)
    
    # 4. Normalize each landmark (transform x,y and scale; optionally scale z similarly)
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
    
    # 5. Placeholder for court normalization (can be replaced with YOLO-based detection)
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
    Dummy function for court normalization.
    Returns a court-relative position based on the hip center.
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
    data: numpy array of shape (n_frames, n_features)
    Returns: derivative computed along the time axis (velocity).
    """
    return np.gradient(data, axis=0) * fps

def smooth_landmark_trajectories(frames, window=15, order=3):
    """
    Smooths the landmark trajectories using the Savitzky–Golay filter.
    frames: numpy array of shape (n_frames, n_features)
    """
    from scipy.signal import savgol_filter
    return savgol_filter(frames, window_length=window, polyorder=order, axis=0)

def process_temporal_features(frames, fps=30):
    """
    Given a list of frames (each a flattened array of normalized landmarks),
    smooth the trajectory and compute its derivative (velocity) and acceleration.
    Returns a dict with keys 'positions', 'velocity', and 'acceleration'.
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
def detect_swing_phase(normalized, temporal):
    """
    Simple swing phase detection using vertical (y-axis) velocity of landmark 16 (right wrist).
    Returns 'backswing' if y-velocity > 0.5, 'forward_swing' if y-velocity < -0.5, else 'neutral'.
    """
    try:
        vel_last = temporal['velocity'][-1]
        idx = 16 * 3 + 1  # For landmark 16, y-value position in flattened array
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
    Flattens a dictionary of normalized landmarks (index -> [x, y, z])
    into a single list in order of landmark indices 0 to 32.
    """
    flat = []
    for i in range(33):
        vals = normalized_landmarks.get(i, [np.nan, np.nan, np.nan])
        flat.extend(vals)
    return flat

def process_csv(input_path, output_path, fps=30):
    """
    Processes a raw pose CSV file, applies spatial normalization to each frame,
    computes temporal features over a sliding window, and writes a new CSV file
    that preserves the original data plus additional columns:
       norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase.
    """
    with open(input_path, 'r') as infile:
        reader = csv.reader(infile)
        original_header = next(reader)
        raw_rows = list(reader)
    
    output_data = []
    frame_buffer = []  # For accumulating flattened normalized landmarks over a window
    temporal_features = None

    for row in raw_rows:
        norm_info = normalize_pose(row)
        frame_buffer.append(flatten_normalized_landmarks(norm_info['normalized_landmarks']))
        if len(frame_buffer) >= TEMPORAL_WINDOW:
            temporal_features = process_temporal_features(frame_buffer, fps=fps)
            frame_buffer.pop(0)  # slide the window
        swing_phase = detect_swing_phase(norm_info, temporal_features) if temporal_features is not None else 'neutral'
        hip_center = norm_info['hip_center']
        court_pos = norm_info['court_position']
        torso_length = norm_info['torso_length']
        extra = [hip_center[0], hip_center[1],
                 court_pos['court_x'], court_pos['court_y'],
                 torso_length, swing_phase]
        output_data.append(row + [str(e) for e in extra])
    
    new_header = original_header + ['norm_hip_x', 'norm_hip_y', 'court_x', 'court_y', 'torso_length', 'swing_phase']
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(output_data)
    print(f"Normalized data saved to {output_path}")

def process_all_csv_files(input_folder, output_folder, fps=30):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('_data.csv')]
    if not csv_files:
        print(f"No CSV files ending with '_data.csv' found in {input_folder}!")
        return
    # Sort files by numeric index extracted from filename (e.g., "video_1_data.csv")
    csv_files = sorted(csv_files, key=lambda f: int(re.search(r'video_(\d+)_data\.csv', f).group(1)))
    for csv_file in csv_files:
        input_csv = os.path.join(input_folder, csv_file)
        output_csv = os.path.join(output_folder, csv_file.replace('_data.csv', '_normalized.csv'))
        print(f"Processing {input_csv} -> {output_csv}")
        process_csv(input_csv, output_csv, fps=fps)

#############################################
# MAIN EXECUTION
#############################################
if __name__ == '__main__':
    process_all_csv_files(INPUT_FOLDER, OUTPUT_FOLDER, fps=FPS)
