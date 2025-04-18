#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_single_clip_hardcoded.py

Processes a single MP4 + optional CSV to produce up to three outputs:
  1) _raw.mp4       (just re-encode)
  2) _overlay.mp4   (raw + skeleton overlaid), if a CSV is provided
  3) _skeleton.mp4  (skeleton on black), if a CSV is provided

No folder scanning, no command-line args—just edit the paths below, run once.

DEPENDENCIES:
    pip install opencv-python numpy
"""

import os
import cv2
import csv
import numpy as np

###################################
# USER SETTINGS (HARD-CODED)
###################################
VIDEO_PATH = "data/personal/Video Trimmer Cut Video.mp4"  # path to your .mp4
CSV_PATH   = "data/personal/Video Trimmer Cut Video_data.csv"  # or None if you don't have landmarks
DO_RAW      = False    # produce a _raw.mp4?
DO_OVERLAY  = True    # produce an _overlay.mp4 (requires CSV)?
DO_SKELETON = True    # produce a _skeleton.mp4 (requires CSV)?

# Skeleton config:
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

COLOR_LEFT = (255, 0, 0)    
COLOR_RIGHT = (0, 0, 255)   
COLOR_CENTER = (0, 255, 0)  
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

# If you want partial transparency in overlay mode, set ALPHA < 1.0 
# e.g. ALPHA=0.7 => 70% raw, 30% skeleton
ALPHA = 1.0  

###################################
# HELPER FUNCTIONS
###################################
def load_landmarks(csv_path):
    """
    Reads CSV, returns a list of dict: frames[i][lm_id] = (x, y) in [0..1].
    Expects columns like 'lm_0_x', 'lm_0_y', etc.
    """
    frames = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Figure out which columns store each of the 33 landmarks
        lm_xy = {}
        for lm_id in range(33):
            x_col = f"lm_{lm_id}_x"
            y_col = f"lm_{lm_id}_y"
            if x_col in header and y_col in header:
                x_idx = header.index(x_col)
                y_idx = header.index(y_col)
                lm_xy[lm_id] = (x_idx, y_idx)

        for row in reader:
            landm = {}
            for lm_id in range(33):
                if lm_id in lm_xy:
                    x_c, y_c = lm_xy[lm_id]
                    x_val = float(row[x_c])
                    y_val = float(row[y_c])
                    landm[lm_id] = (x_val, y_val)
                else:
                    landm[lm_id] = (0.0, 0.0)
            frames.append(landm)
    return frames

def draw_pretty_skeleton(frame_bgr, landmarks_dict, width, height):
    """
    Draws a color-coded skeleton with thicker lines and anti-aliasing.
    """
    # Center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA * width), int(yA * height))
        ptB = (int(xB * width), int(yB * height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA * width), int(yA * height))
        ptB = (int(xB * width), int(yB * height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA * width), int(yA * height))
        ptB = (int(xB * width), int(yB * height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id in range(33):
        nx, ny = landmarks_dict[lm_id]
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

def process_mp4(video_path, csv_path, mode="raw"):
    """
    - "raw": re-encode video as is
    - "overlay": draws skeleton over raw
    - "skeleton": draws skeleton on black
    If output file exists, skip it.
    Returns out_path or None if skipping.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.dirname(video_path)
    suffix = f"_{mode}.mp4"
    out_name = base + suffix
    out_path = os.path.join(folder, out_name)

    # If file already exists, skip
    if os.path.isfile(out_path):
        print(f"[SKIP] {out_name} exists. Not overwriting.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[INFO] Creating => {out_path} (mode={mode})")

    # load CSV if needed
    frames_landmarks = []
    if mode in ["overlay","skeleton"] and csv_path is not None:
        if not os.path.isfile(csv_path):
            print(f"[WARN] CSV not found: {csv_path}, skipping {mode}")
            cap.release()
            out.release()
            os.remove(out_path)  # remove partial
            return None
        frames_landmarks = load_landmarks(csv_path)
        print(f"[INFO] Found {len(frames_landmarks)} frames in {csv_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "raw":
            # just copy frames
            out.write(frame)

        elif mode == "skeleton":
            # black background
            black_img = np.zeros((height, width, 3), dtype=np.uint8)
            if frame_idx < len(frames_landmarks):
                draw_pretty_skeleton(black_img, frames_landmarks[frame_idx], width, height)
            out.write(black_img)

        elif mode == "overlay":
            if frame_idx < len(frames_landmarks):
                if ALPHA < 1.0:
                    overlay = frame.copy()
                    draw_pretty_skeleton(overlay, frames_landmarks[frame_idx], width, height)
                    blend_factor = 1.0 - ALPHA
                    cv2.addWeighted(overlay, blend_factor, frame, ALPHA, 0, frame)
                else:
                    draw_pretty_skeleton(frame, frames_landmarks[frame_idx], width, height)
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] => {out_path}\n")
    return out_path

def main():
    # 1) Check if video file exists
    if not os.path.isfile(VIDEO_PATH):
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        return

    # 2) RAW
    if DO_RAW:
        process_mp4(VIDEO_PATH, csv_path=None, mode="raw")

    # 3) OVERLAY
    if DO_OVERLAY and CSV_PATH is not None:
        process_mp4(VIDEO_PATH, csv_path=CSV_PATH, mode="overlay")

    # 4) SKELETON
    if DO_SKELETON and CSV_PATH is not None:
        process_mp4(VIDEO_PATH, csv_path=CSV_PATH, mode="skeleton")

if __name__ == "__main__":
    main()