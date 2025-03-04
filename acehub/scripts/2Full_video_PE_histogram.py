#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2Full_video_PE_histogram.py

This script processes every video folder in the "data" directory.
For each subfolder (e.g., "video_1") containing an MP4 file,
it runs MediaPipe Pose frame-by-frame and saves the raw pose data
(33 landmarks + optional elbow angles) into a CSV file in the same folder.
"""

import os
import re
import csv
import math
import cv2
import mediapipe as mp

#############################################
# CONFIGURATION SETTINGS
#############################################
# Determine the script directory and set BASE_DATA_DIR relative to it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")  # Adjust as needed

VIDEO_FOLDER_PREFIX = "video_"  # Each video folder is named like "video_1", "video_2", etc.
CALCULATE_ANGLES = True         # Set to True to calculate elbow angles

# Set up MediaPipe Pose (BlazePose Full)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Use 2 for full-body (BlazePose Full)
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#############################################
# HELPER FUNCTIONS
#############################################
def find_video_file(folder):
    """
    Searches for an MP4 file in the given folder.
    1) Looks for a file that exactly matches the folder name (e.g., "video_1.mp4").
    2) Otherwise returns the first file ending with .mp4.
    """
    folder_name = os.path.basename(folder)
    expected_filename = f"{folder_name}.mp4"
    for filename in os.listdir(folder):
        if filename == expected_filename:
            return filename
    for filename in os.listdir(folder):
        if filename.lower().endswith(".mp4"):
            return filename
    return None

def get_video_number(filename):
    """
    Extracts a numeric index from a filename like 'video_10.mp4' => 10 for sorting.
    """
    match = re.search(r'video_(\d+)\.mp4', filename)
    return int(match.group(1)) if match else 999999

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

def process_video(video_path, output_csv):
    """
    Opens the video at `video_path`, runs MediaPipe Pose on each frame,
    and writes the raw pose data to `output_csv`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    print(f"[INFO] Processing {os.path.basename(video_path)} -> {os.path.basename(output_csv)}")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Build header: frame_index + 33 landmarks (x,y,z,vis) each + optional angles
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        if CALCULATE_ANGLES:
            header += ["right_elbow_angle", "left_elbow_angle"]
        writer.writerow(header)
        print(f"[INFO] Header written: {header}")

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                row = [frame_index]
                landmarks = results.pose_landmarks.landmark
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                if CALCULATE_ANGLES:
                    r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                    l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]
                    r_angle = calculate_angle(r_shoulder.x, r_shoulder.y,
                                              r_elbow.x,    r_elbow.y,
                                              r_wrist.x,    r_wrist.y)
                    l_angle = calculate_angle(l_shoulder.x, l_shoulder.y,
                                              l_elbow.x,    l_elbow.y,
                                              l_wrist.x,    l_wrist.y)
                    row += [r_angle, l_angle]
                writer.writerow(row)
            frame_index += 1

    cap.release()
    print(f"[INFO] Done processing {os.path.basename(video_path)}")

#############################################
# MAIN EXECUTION
#############################################
def main():
    if not os.path.isdir(BASE_DATA_DIR):
        print(f"[ERROR] Base data directory not found: {BASE_DATA_DIR}")
        return

    # Iterate through each subfolder in BASE_DATA_DIR that starts with VIDEO_FOLDER_PREFIX
    for folder_name in os.listdir(BASE_DATA_DIR):
        if not folder_name.startswith(VIDEO_FOLDER_PREFIX):
            continue
        video_folder = os.path.join(BASE_DATA_DIR, folder_name)
        if not os.path.isdir(video_folder):
            continue

        video_file = find_video_file(video_folder)
        if video_file is None:
            print(f"[WARNING] No .mp4 file found in {video_folder}. Skipping.")
            continue

        video_path = os.path.join(video_folder, video_file)
        video_basename = os.path.splitext(video_file)[0]
        # Output CSV saved in the same folder as the video file
        output_csv = os.path.join(video_folder, f"{video_basename}_data.csv")
        if os.path.exists(output_csv):
            os.remove(output_csv)
        process_video(video_path, output_csv)

if __name__ == "__main__":
    main()
