#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
all_in_one_pre_cut.py

Combines three stages in one script:
  1) Acquisition     - Download & convert YouTube videos to 30 FPS
  2) Pose Extraction - Run MediaPipe Pose to produce raw CSVs
  3) Normalization   - Spatially & temporally normalize each CSV

Directory usage:
  data/raw/
      video_X/
          video_X.mp4   (final MP4 after download & conversion)
  data/interim/
      video_X/
          video_X_data.csv
  data/processed/
      video_X/
          video_X_normalized.csv

Simply edit `youtube_urls` as needed, then run:
    python all_in_one_pre_cut.py
"""

import os
import sys
import math
import csv
import shutil
import cv2
import ffmpeg
import mediapipe as mp
import yt_dlp
from multiprocessing import Pool, cpu_count
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
import numpy as np

# -------------------
# USER CONFIG
# -------------------
# Provide the full list of YouTube links here:
youtube_urls = [
    # Example link:
    "https://www.youtube.com/watch?v=NxeIZZjyAkM&ab_channel=COURTLEVELTENNIS-LiamApilado"
    # Add more links as desired
]

FPS = 30              # Desired frame rate for final MP4
CALCULATE_ANGLES = True
FORCE_REPROCESS = False  # If True, re-extract & re-normalize even if CSVs exist

# Maximum parallel workers for pose extraction:
NUM_PROCESSES = min(8, cpu_count())

# Root directories (relative paths)
# Adjust if your structure differs
RAW_DIR       = os.path.join("data", "raw")      # final MP4 goes here
INTERIM_DIR   = os.path.join("data", "interim")  # raw CSV (pose data)
PROCESSED_DIR = os.path.join("data", "processed")# normalized CSV

# -------------------------------------------------------------------------
# PART 1: ACQUISITION - Download & Convert to 30 FPS
# -------------------------------------------------------------------------
def download_video(url, idx, video_folder):
    """
    Downloads best MP4 from YouTube using yt-dlp, storing as
    e.g. "video_3_temp.mp4". Returns path to temp file.
    """
    temp_filename = f"video_{idx}_temp.mp4"
    outtmpl = os.path.join(video_folder, temp_filename)
    ydl_opts = {
    # Instead of the old format string "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
    # we allow a more flexible fallback:
    "format": "bv*+ba/best",  
    "merge_output_format": "mp4",   # will still try merging to an MP4 container
    "outtmpl": outtmpl,
    "noplaylist": True,
    # You can optionally set "ignoreerrors": True,
    # or "skip_unavailable_fragments": True if needed
}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"[INFO] Downloaded temporary video: {outtmpl}")
    return outtmpl

def convert_video_to_30fps(input_path, output_path, fps):
    """
    Converts a raw MP4 to the given FPS using FFmpeg; re-encodes video with libx264 and copies audio.
    """
    try:
        stream = ffmpeg.input(input_path)
        stream = stream.filter('fps', fps=fps, round='up')
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec='libx264',
            crf=23,
            preset='fast',
            acodec='copy'
        )
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        print(f"[INFO] Converted => {output_path} @ {fps} FPS")
    except ffmpeg.Error as e:
        print(f"[ERROR] FFmpeg conversion failed on {input_path}: {e}")

def find_next_available_index(directory):
    """
    Finds the next available 'video_X' index within the target directory.
    E.g. if video_1, video_2 already exist, returns 3.
    """
    max_idx = 0
    for item in os.listdir(directory):
        if item.startswith("video_") and os.path.isdir(os.path.join(directory, item)):
            try:
                idx = int(item.split("_")[1])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                pass
    return max_idx + 1

def run_acquisition():
    """
    Downloads each YouTube video to data/raw/video_X,
    converts to 30 FPS as video_X.mp4, removing the temp file afterwards.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    start_index = find_next_available_index(RAW_DIR)
    print(f"\n=== ACQUISITION: Checking from video index {start_index} ===")

    for i, url in enumerate(youtube_urls):
        idx = start_index + i
        print(f"[INFO] Downloading video_{idx}: {url}")

        video_folder = os.path.join(RAW_DIR, f"video_{idx}")
        final_path   = os.path.join(video_folder, f"video_{idx}.mp4")

        if os.path.isfile(final_path):
            print(f"[SKIP] video_{idx}.mp4 already exists => {final_path}")
            continue

        os.makedirs(video_folder, exist_ok=True)

        # 1) Download
        temp_path = download_video(url, idx, video_folder)
        # 2) Convert to 30 FPS
        convert_video_to_30fps(temp_path, final_path, FPS)
        # 3) Remove temp
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[INFO] Removed temp file: {temp_path}")
    print("=== ACQUISITION COMPLETE ===\n")

# -------------------------------------------------------------------------
# PART 2: POSE EXTRACTION - MediaPipe -> data/interim/...
# -------------------------------------------------------------------------
def find_mp4_in_raw_folder(folder):
    """
    e.g. folder='video_3' => expects 'video_3.mp4'
    or else the first .mp4 found.
    """
    # either exactly folder_name.mp4 or any mp4
    folder_name = os.path.basename(folder)
    mp4_expected = f"{folder_name}.mp4"
    for fname in os.listdir(folder):
        if fname == mp4_expected:
            return os.path.join(folder, fname)

    for fname in os.listdir(folder):
        if fname.lower().endswith(".mp4"):
            return os.path.join(folder, fname)
    return None

def extract_angles(landmarks, row):
    # angles for R/L elbow
    # right side: 12=Rshoulder,14=Relbow,16=Rwrist
    rs, re, rw = landmarks[12], landmarks[14], landmarks[16]
    r_angle = calculate_angle_2d(rs.x, rs.y, re.x, re.y, rw.x, rw.y)
    # left side: 11=Lshoulder,13= Lelbow,15=LWrist
    ls, le, lw = landmarks[11], landmarks[13], landmarks[15]
    l_angle = calculate_angle_2d(ls.x, ls.y, le.x, le.y, lw.x, lw.y)
    row += [r_angle, l_angle]

def calculate_angle_2d(ax, ay, bx, by, cx, cy):
    """
    Returns angle at B formed by A->B->C in 2D space.
    """
    BAx = ax - bx
    BAy = ay - by
    BCx = cx - bx
    BCy = cy - by
    dot = (BAx * BCx) + (BAy * BCy)
    magBA = math.sqrt(BAx**2 + BAy**2)
    magBC = math.sqrt(BCx**2 + BCy**2)
    if magBA == 0 or magBC == 0:
        return 0.0
    cos_angle = max(min(dot / (magBA * magBC), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def process_single_video_pose(video_num):
    """
    1) Finds data/raw/video_num/video_num.mp4
    2) Creates data/interim/video_num folder => store video_num_data.csv
    3) Uses MediaPipe Pose on each frame
    """
    raw_folder = os.path.join(RAW_DIR, f"video_{video_num}")
    mp4_path   = os.path.join(raw_folder, f"video_{video_num}.mp4")
    if not os.path.isfile(mp4_path):
        print(f"[SKIP] No MP4 => {mp4_path}")
        return

    # The output CSV is stored in data/interim/video_{video_num}/video_{video_num}_data.csv
    out_folder = os.path.join(INTERIM_DIR, f"video_{video_num}")
    os.makedirs(out_folder, exist_ok=True)
    out_csv = os.path.join(out_folder, f"video_{video_num}_data.csv")

    if os.path.exists(out_csv) and not FORCE_REPROCESS:
        print(f"[SKIP] Already have => {out_csv}")
        return

    print(f"[EXTRACT] {mp4_path} -> {out_csv}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        smooth_landmarks=True, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open => {mp4_path}")
        return

    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # build header
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        if CALCULATE_ANGLES:
            header += ["right_elbow_angle", "left_elbow_angle"]
        writer.writerow(header)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                row = [frame_idx]
                lms = results.pose_landmarks.landmark
                for lm in lms:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                if CALCULATE_ANGLES:
                    extract_angles(lms, row)
                writer.writerow(row)
            frame_idx += 1

    cap.release()
    print(f"[DONE] Pose => {out_csv}")

def run_pose_extraction():
    """
    Loop through data/raw for all 'video_X' folders and do process_single_video_pose(X).
    """
    os.makedirs(INTERIM_DIR, exist_ok=True)

    # find all subfolders 'video_#'
    for item in os.listdir(RAW_DIR):
        if not item.startswith("video_"):
            continue
        try:
            idx = int(item.split("_")[1])
        except:
            continue
        process_single_video_pose(idx)
    print("=== POSE EXTRACTION COMPLETE ===\n")

# -------------------------------------------------------------------------
# PART 3: NORMALIZATION - data/interim => data/processed
# -------------------------------------------------------------------------
def normalize_pose(row):
    """
    Takes one CSV row => 1 + 33*4 columns => normalize to
    (hip-centered + rotated + scaled by torso_length).
    """
    landmarks = []
    # row[0]=frame_index; row[1..]=landmarks
    for i in range(33):
        base = 1 + (i*4)
        x = float(row[base])
        y = float(row[base+1])
        z = float(row[base+2])
        vis = float(row[base+3])
        landmarks.append((x, y, z, vis))

    # compute hip center (23,24)
    hx, hy, hz, _ = landmarks[23]
    hx2, hy2, hz2, _ = landmarks[24]
    hip_center = np.array([ (hx+hx2)/2.0, (hy+hy2)/2.0, (hz+hz2)/2.0 ])

    # shoulder center (11,12)
    sx, sy, sz, _ = landmarks[11]
    sx2, sy2, sz2, _ = landmarks[12]
    shoulder_center = np.array([ (sx+sx2)/2.0, (sy+sy2)/2.0, (sz+sz2)/2.0 ])

    torso_vec = shoulder_center - hip_center
    rotation_angle = math.atan2(torso_vec[1], torso_vec[0])  # y,x
    rot_mat = Rotation.from_euler('z', -rotation_angle).as_matrix()[0:2, 0:2]
    torso_len = np.linalg.norm(torso_vec)

    # normalize each landmark
    normalized_dict = {}
    for idx, (lx, ly, lz, lvis) in enumerate(landmarks):
        if lvis < 0.5:  # VISIBILITY_THRESHOLD
            normalized_dict[idx] = [np.nan, np.nan, np.nan]
            continue

        # shift => rotate => scale
        shift_xy = np.array([lx, ly]) - hip_center[:2]
        rot_xy   = rot_mat @ shift_xy
        scale_xy = rot_xy / torso_len if torso_len != 0 else rot_xy
        z_rel    = (lz - hip_center[2]) / torso_len if torso_len != 0 else (lz - hip_center[2])

        normalized_dict[idx] = [scale_xy[0], scale_xy[1], z_rel]

    # example "court" pos
    court_x = hip_center[0] * 1.5
    court_y = hip_center[1] * 0.8

    return {
        "norm_lm": normalized_dict,
        "torso_len": torso_len,
        "hip_center": hip_center,
        "court_x": court_x,
        "court_y": court_y
    }

def flatten_norm_landmarks(norm_map):
    """
    norm_map is { idx->[x,y,z], ... }, flatten to length 99
    """
    flat = []
    for i in range(33):
        val = norm_map.get(i, [np.nan,np.nan,np.nan])
        flat.extend(val)
    return flat

def detect_swing_phase(norm_info, velocity_data):
    """
    Example velocity-based phase detection: uses LM16 (right wrist) velocity Y
    to see if it's positive => backswing, negative => forward_swing, else neutral.
    We'll skip if no velocity_data.
    """
    if velocity_data is None:
        return "neutral"
    # velocity_data is shape (frames, 99?), we look at the last frame in that window
    # LM16 => indices [48..50], y => 49
    last_vel = velocity_data[-1]
    vy = last_vel[49]
    if vy > 0.5:
        return "backswing"
    elif vy < -0.5:
        return "forward_swing"
    return "neutral"

def smooth_landmark_trajectories(frames_arr):
    """
    frames_arr => shape (N, 99)
    applies Savitzky-Golay
    """
    if len(frames_arr) < 15:
        # not enough frames, just return
        return np.array(frames_arr)
    return savgol_filter(frames_arr, window_length=15, polyorder=3, axis=0)

def derivative(data, fps=30):
    return np.gradient(data, axis=0)*fps

def process_single_csv_norm(input_csv, output_csv, fps=30):
    """
    1) read raw csv => for each row => normalize
    2) keep a sliding window of length 30 for velocity detection
    3) detect swing phase => store columns
    """
    rows = []
    with open(input_csv, "r") as f:
        rd = csv.reader(f)
        header = next(rd)
        for line in rd:
            rows.append(line)

    out_data = []
    frame_buffer = []  # store flattened frames for temporal
    for row in rows:
        # row[0]=frame_idx
        # row[1..] => 33*4 => x,y,z,vis
        ni = normalize_pose(row)
        # flatten
        flat = flatten_norm_landmarks(ni["norm_lm"])
        frame_buffer.append(flat)

        velocity_window = None
        if len(frame_buffer) >= 30:
            # we do smoothing => velocity => remove oldest
            arr = np.array(frame_buffer)
            arr_smooth = smooth_landmark_trajectories(arr)
            vel = derivative(arr_smooth, fps=fps)
            velocity_window = vel
            frame_buffer.pop(0)
        phase = detect_swing_phase(ni, velocity_window)

        # extras
        extra = [
            ni["hip_center"][0], ni["hip_center"][1],
            ni["court_x"], ni["court_y"],
            ni["torso_len"], phase
        ]
        out_data.append(row + [str(v) for v in extra])

    new_header = rows[0][:1] + rows[0][1:]  # same as original, but let's be explicit
    # Actually we want to keep the original header + new columns:
    # new columns: norm_hip_x, norm_hip_y, court_x, court_y, torso_length, swing_phase
    # We'll just do:
    with open(output_csv, "w", newline="") as of:
        wr = csv.writer(of)
        # Re-build the original header + new
        # The original had frame_index + 33*(x,y,z,vis)
        # So let's replicate that approach:
        orig_header = ["frame_index"]
        for i in range(33):
            orig_header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        if len(rows[0]) == 1 + 33*4 + 2:  
            # if the input had angles, etc. -> you'd adapt here
            pass

        final_header = orig_header + [
            "norm_hip_x","norm_hip_y","court_x","court_y","torso_length","swing_phase"
        ]
        wr.writerow(final_header)
        wr.writerows(out_data)
    print(f"[NORMALIZED] => {output_csv}")

def run_normalization():
    """
    For each video_X folder in data/interim,
    find "video_X_data.csv" => produce "video_X_normalized.csv" in data/processed/video_X
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for folder_name in os.listdir(INTERIM_DIR):
        if not folder_name.startswith("video_"):
            continue
        subdir = os.path.join(INTERIM_DIR, folder_name)
        if not os.path.isdir(subdir):
            continue

        # find *_data.csv
        data_csv = None
        for f in os.listdir(subdir):
            if f.endswith("_data.csv"):
                data_csv = os.path.join(subdir, f)
                break
        if not data_csv:
            print(f"[SKIP] No data CSV in {subdir}")
            continue
        # build output dir in data/processed
        out_dir = os.path.join(PROCESSED_DIR, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{folder_name}_normalized.csv")

        if os.path.isfile(out_csv) and not FORCE_REPROCESS:
            print(f"[SKIP] normalized => {out_csv}")
            continue

        print(f"[NORM] {data_csv} => {out_csv}")
        process_single_csv_norm(data_csv, out_csv, fps=FPS)

    print("=== NORMALIZATION COMPLETE ===\n")


# -------------------------------------------------------------------------
# MASTER "MAIN" - calls all steps
# -------------------------------------------------------------------------
def main():
    """
    1) run_acquisition - download & convert videos to data/raw/video_X/video_X.mp4
    2) run_pose_extraction - produce data/interim/video_X/video_X_data.csv
    3) run_normalization - produce data/processed/video_X/video_X_normalized.csv
    """
    # Step A: Download + convert
    run_acquisition()

    # Step B: Pose extraction => data/interim
    run_pose_extraction()

    # Step C: Normalization => data/processed
    run_normalization()

    print("All-in-one pre-cut pipeline is DONE.\n"
          "You can now manually create .llc for each video_x and run the post-cut scripts if needed.")


if __name__ == "__main__":
    main()
