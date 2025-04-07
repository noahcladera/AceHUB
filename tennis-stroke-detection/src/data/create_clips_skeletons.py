#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_process_final_library.py

Scans "Final library/" for MP4 files, checks if there's a matching CSV,
and produces up to three new MP4s per clip:
  1) _raw.mp4       (just re-encode)
  2) _overlay.mp4   (raw + skeleton), if CSV found
  3) _skeleton.mp4  (skeleton on black), if CSV found

NOW with a check to see if each output file already exists.
If it does, we skip regenerating that file.

DEPENDENCIES:
    pip install opencv-python numpy
"""

import os
import cv2
import csv
import numpy as np

FINAL_LIBRARY = "Final library"

# -------------- SKELETON CONFIG --------------
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

COLOR_LEFT = (255, 0, 0)    # Blue BGR
COLOR_RIGHT = (0, 0, 255)   # Red
COLOR_CENTER = (0, 255, 0)  # Green
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

# If you want partial transparency in overlay mode, set ALPHA < 1.0 (e.g. 0.7)
ALPHA = 1.0

def load_landmarks(csv_path):
    """
    Reads the CSV and returns a list of dict frames[i][lm_id] = (x, y) in [0..1].
    We assume columns named 'lm_0_x', 'lm_0_y', etc.
    """
    frames = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Identify columns for each landmark
        lm_xy = {}
        for lm_id in range(33):
            x_col_name = f"lm_{lm_id}_x"
            y_col_name = f"lm_{lm_id}_y"
            if x_col_name in header and y_col_name in header:
                x_idx = header.index(x_col_name)
                y_idx = header.index(y_col_name)
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
    Draw a color-coded skeleton with thicker lines and anti-aliasing.
    Blue (left), Red (right), Green (center).
    """
    # Draw center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA*width), int(yA*height))
        ptB = (int(xB*width), int(yB*height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER,
                 thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA*width), int(yA*height))
        ptB = (int(xB*width), int(yB*height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT,
                 thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        xA, yA = landmarks_dict[lmA]
        xB, yB = landmarks_dict[lmB]
        ptA = (int(xA*width), int(yA*height))
        ptB = (int(xB*width), int(yB*height))
        cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT,
                 thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id in range(33):
        nx, ny = landmarks_dict[lm_id]
        px = int(nx*width)
        py = int(ny*height)
        # color-coded keypoints or center color
        if lm_id in (11,13,15,23,25,27):
            kp_color = COLOR_LEFT
        elif lm_id in (12,14,16,24,26,28):
            kp_color = COLOR_RIGHT
        else:
            kp_color = COLOR_CENTER

        cv2.circle(frame_bgr, (px, py), CIRCLE_RADIUS, kp_color, -1, lineType=LINE_TYPE)

def process_mp4(video_path, csv_path=None, mode="overlay"):
    """
    - If mode="overlay": raw + skeleton (requires csv)
    - If mode="skeleton": skeleton on black (requires csv)
    - If mode="raw": copy/reencode
    Returns the name of the new MP4 or None if error or if it already exists.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    dirpath = os.path.dirname(video_path)

    suffix = f"_{mode}.mp4"
    out_name = basename + suffix
    out_path = os.path.join(dirpath, out_name)

    # 1) If the output already exists, skip
    if os.path.isfile(out_path):
        print(f"[SKIP] {out_name} already exists. Not overwriting.")
        return None

    # 2) Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"[INFO] Creating => {out_path} (mode={mode})")

    frames_landmarks = []
    if mode in ["overlay","skeleton"] and csv_path:
        if not os.path.isfile(csv_path):
            print(f"[WARN] No CSV found: {csv_path} => skip {mode}")
            cap.release()
            out.release()
            os.remove(out_path)  # remove partial file
            return None
        frames_landmarks = load_landmarks(csv_path)
        print(f"[INFO] Loaded {len(frames_landmarks)} frames from CSV => {csv_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "raw":
            out.write(frame)

        elif mode == "skeleton":
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            if frame_idx < len(frames_landmarks):
                draw_pretty_skeleton(blank, frames_landmarks[frame_idx], width, height)
            out.write(blank)

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
    print(f"Scanning folder: {FINAL_LIBRARY}")

    # Loop over everything in Final library
    for fname in os.listdir(FINAL_LIBRARY):
        if not fname.lower().endswith(".mp4"):
            continue
        video_path = os.path.join(FINAL_LIBRARY, fname)
        base = os.path.splitext(fname)[0]
        # The expected CSV is "base.csv"
        csv_path = os.path.join(FINAL_LIBRARY, base + ".csv")
        print(f"\n[PROCESSING] {fname}")


        # 2) overlay, if CSV is found
        if os.path.isfile(csv_path):
            process_mp4(video_path, csv_path=csv_path, mode="overlay")

        # 3) skeleton-only, if CSV is found
        if os.path.isfile(csv_path):
            process_mp4(video_path, csv_path=csv_path, mode="skeleton")

if __name__ == "__main__":
    main()
