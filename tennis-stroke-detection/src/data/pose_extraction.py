#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pose_extraction.py

Processes each raw video in "tennis-stroke-detection/data/raw" by running
MediaPipe Pose frame-by-frame to produce a CSV of raw pose data.
This version supports multiprocessing and skips already-processed videos
unless FORCE_REPROCESS is set to True.
"""

import os
import re
import csv
import math
import cv2
import mediapipe as mp
from multiprocessing import Pool, cpu_count

#############################################
# CONFIGURATION SETTINGS
#############################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The raw videos are assumed in data/raw, and we keep CSVs in the same folder or a subfolder
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "interim")

VIDEO_FOLDER_PREFIX = "video_"
CALCULATE_ANGLES = True
FORCE_REPROCESS = False

NUM_PROCESSES = min(8, cpu_count())  # or adjust as desired

#############################################
# HELPER FUNCTIONS
#############################################
def find_video_file(folder):
    """
    Searches for an MP4 file in the given folder:
      1) Checking for a file that exactly matches the folder name (e.g., "video_1.mp4").
      2) Otherwise returns the first .mp4 file found.
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
    Extracts an integer from names like 'video_10.mp4' => 10 (for sorting).
    """
    match = re.search(r'video_(\d+)\.mp4', filename)
    return int(match.group(1)) if match else 999999

def calculate_angle(ax, ay, bx, by, cx, cy):
    """
    Calculate angle at point B formed by A->B->C in 2D space.
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

def process_video_task(task_data):
    """
    Worker function for parallel processing.
    Each task_data is (video_path, output_csv).
    """
    video_path, output_csv = task_data

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
        print(f"[ERROR] Could not open video file: {video_path}")
        return None

    print(f"[INFO] Processing: {os.path.basename(video_path)} -> {os.path.basename(output_csv)}")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        if CALCULATE_ANGLES:
            header += ["right_elbow_angle", "left_elbow_angle"]
        writer.writerow(header)

        frame_index = 0
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
    print(f"[INFO] Finished: {os.path.basename(video_path)}")
    return os.path.basename(video_path)

#############################################
# MAIN EXECUTION
#############################################
def main():
    if not os.path.isdir(BASE_DATA_DIR):
        print(f"[ERROR] Data folder not found: {BASE_DATA_DIR}")
        return

    tasks = []
    skipped_videos = []

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
        output_csv = os.path.join(video_folder, f"{video_basename}_data.csv")

        if os.path.exists(output_csv) and not FORCE_REPROCESS:
            print(f"[INFO] Already processed: {video_basename}. Skipping.")
            skipped_videos.append(os.path.basename(video_path))
            continue

        tasks.append((video_path, output_csv))

    print(f"[INFO] Starting parallel processing with {NUM_PROCESSES} workers")
    print(f"[INFO] Found {len(tasks)} videos to process")
    print(f"[INFO] Skipped {len(skipped_videos)} videos with existing data.csv")

    if not tasks:
        print("[INFO] No videos to process.")
        return

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_video_task, tasks)
        results = [r for r in results if r is not None]

    print(f"[INFO] Processed videos: {results}")

if __name__ == "__main__":
    main()