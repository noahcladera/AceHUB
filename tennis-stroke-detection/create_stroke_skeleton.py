#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_stroke_skeleton.py

Takes a stroke from the Strokes_Library folder and creates skeleton visualizations.
Can be used with a sample video or just the stroke's CSV data.

Produces up to two outputs:
  1) stroke_XXXX_skeleton.mp4  (skeleton on black background)
  2) stroke_XXXX_overlay.mp4   (skeleton overlaid on original video), if a video is provided

Usage:
  python create_stroke_skeleton.py [stroke_id] [video_path]

  - If stroke_id is not provided, defaults to 1503
  - If video_path is not provided, only creates skeleton animation (no overlay)

DEPENDENCIES:
    pip install opencv-python numpy
"""

import os
import sys
import cv2
import csv
import numpy as np

# Skeleton config:
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

COLOR_LEFT = (255, 0, 0)    # Blue in BGR
COLOR_RIGHT = (0, 0, 255)   # Red in BGR  
COLOR_CENTER = (0, 255, 0)  # Green in BGR
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

# If you want partial transparency in overlay mode, set ALPHA < 1.0 
# e.g. ALPHA=0.7 => 70% raw, 30% skeleton
ALPHA = 1.0  

# Default output dimensions for skeleton-only video
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

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

def create_skeleton_video(landmarks_frames, output_path, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
    """
    Creates a skeleton-only video (black background) from landmarks data
    """
    if os.path.isfile(output_path):
        print(f"[SKIP] {output_path} exists. Not overwriting.")
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Creating skeleton video: {output_path}")
    
    for frame_idx, landmarks in enumerate(landmarks_frames):
        black_img = np.zeros((height, width, 3), dtype=np.uint8)
        draw_pretty_skeleton(black_img, landmarks, width, height)
        out.write(black_img)
    
    out.release()
    print(f"[DONE] => {output_path}")
    return output_path

def create_overlay_video(video_path, landmarks_frames, output_path):
    """
    Creates an overlay video (skeleton on original video) from landmarks data
    """
    if os.path.isfile(output_path):
        print(f"[SKIP] {output_path} exists. Not overwriting.")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Creating overlay video: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(landmarks_frames):
            if ALPHA < 1.0:
                overlay = frame.copy()
                draw_pretty_skeleton(overlay, landmarks_frames[frame_idx], width, height)
                blend_factor = 1.0 - ALPHA
                cv2.addWeighted(overlay, blend_factor, frame, ALPHA, 0, frame)
            else:
                draw_pretty_skeleton(frame, landmarks_frames[frame_idx], width, height)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] => {output_path}")
    return output_path

def main():
    # Parse command line arguments
    stroke_id = "1503"  # default
    video_path = None
    
    if len(sys.argv) > 1:
        stroke_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        video_path = sys.argv[2]
    
    stroke_dir = f"Strokes_Library/stroke_{stroke_id}"
    csv_path = f"{stroke_dir}/stroke.csv"
    
    # Check if CSV exists
    if not os.path.isfile(csv_path):
        print(f"[ERROR] Stroke CSV not found: {csv_path}")
        print(f"Make sure the stroke ID is correct and the file exists.")
        return
    
    # Load landmarks data
    print(f"[INFO] Loading landmarks from {csv_path}")
    landmarks_frames = load_landmarks(csv_path)
    print(f"[INFO] Loaded {len(landmarks_frames)} frames of landmark data")
    
    # Create output directory if it doesn't exist
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create skeleton-only video
    skeleton_output = f"{output_dir}/stroke_{stroke_id}_skeleton.mp4"
    create_skeleton_video(landmarks_frames, skeleton_output)
    
    # Create overlay video if video path is provided
    if video_path:
        if not os.path.isfile(video_path):
            print(f"[ERROR] Video not found: {video_path}")
        else:
            overlay_output = f"{output_dir}/stroke_{stroke_id}_overlay.mp4"
            create_overlay_video(video_path, landmarks_frames, overlay_output)

if __name__ == "__main__":
    main() 