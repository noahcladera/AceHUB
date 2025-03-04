#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2Full_video_PE_histogram.py

This script processes every video folder in the "data" directory.
For each subfolder (e.g., "video_1/") containing an MP4 file,
it runs MediaPipe Pose frame-by-frame and saves the raw pose data
(with 33 landmarks and optionally elbow angles) into a CSV file in the same folder.
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
# Base directory containing video subfolders (each video gets its own folder)
BASE_DATA_DIR = "data"
VIDEO_FOLDER_PREFIX = "video_"  # Each video folder name starts with this
CALCULATE_ANGLES = True         # Set to True to calculate extra joint angles

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
    Searches for the MP4 file in the given folder.
    It first looks for a file that exactly matches the folder name (e.g., "video_1.mp4").
    If not found, it returns the first file ending with .mp4.
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
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # Clamp to avoid domain errors
    return math.degrees(math.acos(cos_angle))

def process_video(video_path, output_csv):
    """
    Opens the video at video_path, runs MediaPipe Pose on every frame,
    and writes the raw pose data to output_csv.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    print(f"Processing {os.path.basename(video_path)} -> {output_csv}")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Build header: frame_index + 33 landmarks (x, y, z, visibility) for each,
        # plus optional elbow angles.
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        if CALCULATE_ANGLES:
            header += ["right_elbow_angle", "left_elbow_angle"]
        writer.writerow(header)
        print("Header written:", header)

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
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
                                              r_elbow.x, r_elbow.y,
                                              r_wrist.x, r_wrist.y)
                    l_angle = calculate_angle(l_shoulder.x, l_shoulder.y,
                                              l_elbow.x, l_elbow.y,
                                              l_wrist.x, l_wrist.y)
                    row += [r_angle, l_angle]
                writer.writerow(row)
            frame_index += 1

    cap.release()
    print(f"Done processing {os.path.basename(video_path)}")

#############################################
# MAIN EXECUTION
#############################################
def main():
    # Loop through each subfolder in BASE_DATA_DIR that starts with VIDEO_FOLDER_PREFIX
    for folder_name in os.listdir(BASE_DATA_DIR):
        if not folder_name.startswith(VIDEO_FOLDER_PREFIX):
            continue
        video_folder = os.path.join(BASE_DATA_DIR, folder_name)
        if not os.path.isdir(video_folder):
            continue

        video_file = find_video_file(video_folder)
        if video_file is None:
            print(f"[WARNING] No video file found in {video_folder}. Skipping.")
            continue

        video_path = os.path.join(video_folder, video_file)
        video_basename = os.path.splitext(video_file)[0]
        # The output CSV is saved inside the same folder with a name like "video_1_data.csv"
        output_csv = os.path.join(video_folder, f"{video_basename}_data.csv")
        # Remove the output CSV if it already exists
        if os.path.exists(output_csv):
            os.remove(output_csv)
        process_video(video_path, output_csv)

if __name__ == "__main__":
    main()
