#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pose_extraction_single_hardcoded.py

Extracts pose data from a single MP4 video using MediaPipe Pose, 
writes the results to a CSV. No command-line arguments, 
just edit VIDEO_PATH and OUTPUT_CSV below.

Dependencies:
  pip install opencv-python mediapipe
"""

import os
import csv
import math
import cv2
import mediapipe as mp

##########################################
# USER SETTINGS: Hard-coded paths
##########################################
VIDEO_PATH = "data/personal/Video Trimmer Cut Video.mp4"  # relative or absolute
OUTPUT_CSV = None            # if None, auto-naming "my_video_data.csv"

CALCULATE_ANGLES = True  # set to False if you don't want elbow angles

##########################################
# HELPER FUNCTIONS
##########################################

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
    Runs MediaPipe Pose on the video, writes a CSV with:
      frame_index, lm_0_x, lm_0_y, lm_0_z, lm_0_vis, ...
      plus optional elbow angles.
    """
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
        return

    print(f"[INFO] Processing: {os.path.basename(video_path)} -> {os.path.basename(output_csv)}")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Build header
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
                    # right side: 12=Rshoulder,14=Relbow,16=Rwrist
                    r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                    # left side: 11=Lshoulder,13=Lelbow,15=LWrist
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


##########################################
# MAIN EXECUTION
##########################################
def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"[ERROR] Video file does not exist: {VIDEO_PATH}")
        return

    if OUTPUT_CSV is None:
        base, ext = os.path.splitext(VIDEO_PATH)
        out_csv = base + "_data.csv"
    else:
        out_csv = OUTPUT_CSV

    process_video(VIDEO_PATH, out_csv)

if __name__ == "__main__":
    main()
