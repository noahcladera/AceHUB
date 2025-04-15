#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
personal_post_processing.py

All-in-one post-processing script for generating clips and visualizations 
from a video that's already been set up in the pipeline.

This script handles:
1. Running the stroke segmentation process to generate clips
2. Creating skeleton overlay and skeleton-only videos
3. Creating animated 3D visualizations of the clips

Usage:
    python personal_post_processing.py video_id [--skip-segmentation] [--skip-skeleton] [--skip-3d]

Example:
    python personal_post_processing.py 75   # Process video_75

Dependencies:
    pip install opencv-python numpy pandas plotly mediapipe
"""

import os
import sys
import argparse
import subprocess
import glob
import csv
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.graph_objects as go

# Constants
VIDEOS_DIR = "videos"
STROKES_LIBRARY = "Strokes_Library"
STROKE_SEGMENTATION_SCRIPT = "src/stroke_segmentation.py"

# MediaPipe Pose connections for skeleton drawing
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

# Color configuration
COLOR_LEFT = (255, 0, 0)  # Blue for left side
COLOR_RIGHT = (0, 0, 255)  # Red for right side
COLOR_CENTER = (0, 255, 0)  # Green for center

# Drawing configuration
LINE_THICKNESS = 4
CIRCLE_RADIUS = 6
LINE_TYPE = cv2.LINE_AA

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def run_stroke_segmentation():
    """
    Run the main stroke segmentation script to generate clips
    """
    print(f"[STEP 1/3] Running stroke segmentation process")
    
    try:
        result = subprocess.run(
            ["python", STROKE_SEGMENTATION_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )
        print("Stroke segmentation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running stroke segmentation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def get_strokes_for_video(video_id):
    """
    Find all strokes in the Strokes Library that came from the specified video
    """
    strokes = []
    
    # Look through all stroke folders
    for stroke_dir in glob.glob(os.path.join(STROKES_LIBRARY, "stroke_*")):
        # Check source_info.txt to see if it's from our video
        source_info_path = os.path.join(stroke_dir, "source_info.txt")
        if os.path.exists(source_info_path):
            with open(source_info_path, 'r') as f:
                content = f.read()
                if f"Source Video: video_{video_id}" in content:
                    strokes.append(os.path.basename(stroke_dir))
    
    return sorted(strokes)

def create_skeleton_videos(video_id, strokes):
    """
    Create skeleton overlay and skeleton-only videos for each stroke
    """
    print(f"[STEP 2/3] Creating skeleton videos for {len(strokes)} strokes from video_{video_id}")
    
    # Process each stroke
    successful = 0
    for stroke in strokes:
        stroke_dir = os.path.join(STROKES_LIBRARY, stroke)
        
        # Check if we have the necessary files
        clip_video = os.path.join(stroke_dir, "stroke_clip.mp4")
        csv_file = os.path.join(stroke_dir, "stroke.csv")
        overlay_output = os.path.join(stroke_dir, "stroke_overlay.mp4")
        skeleton_output = os.path.join(stroke_dir, "stroke_skeleton.mp4")
        
        if not os.path.exists(clip_video) or not os.path.exists(csv_file):
            print(f"[SKIP] Missing source files for {stroke}")
            continue
        
        # If both outputs already exist, skip
        if os.path.exists(overlay_output) and os.path.exists(skeleton_output):
            print(f"[SKIP] Skeleton videos already exist for {stroke}")
            successful += 1
            continue
        
        print(f"Processing {stroke}...")
        
        try:
            # Load landmarks
            landmarks_frames = load_landmarks(csv_file)
            
            # Process the video
            if not os.path.exists(overlay_output) or not os.path.exists(skeleton_output):
                create_skeleton_video(clip_video, landmarks_frames, overlay_output, skeleton_output)
                successful += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {stroke}: {e}")
    
    print(f"[DONE] Created skeleton videos for {successful}/{len(strokes)} strokes")
    return successful > 0

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

def draw_skeleton(frame_bgr, landmarks_dict, width, height):
    """
    Draws a color-coded skeleton with thicker lines and anti-aliasing.
    """
    # Center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        if lmA in landmarks_dict and lmB in landmarks_dict:
            xA, yA = landmarks_dict[lmA]
            xB, yB = landmarks_dict[lmB]
            ptA = (int(xA * width), int(yA * height))
            ptB = (int(xB * width), int(yB * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        if lmA in landmarks_dict and lmB in landmarks_dict:
            xA, yA = landmarks_dict[lmA]
            xB, yB = landmarks_dict[lmB]
            ptA = (int(xA * width), int(yA * height))
            ptB = (int(xB * width), int(yB * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        if lmA in landmarks_dict and lmB in landmarks_dict:
            xA, yA = landmarks_dict[lmA]
            xB, yB = landmarks_dict[lmB]
            ptA = (int(xA * width), int(yA * height))
            ptB = (int(xB * width), int(yB * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id in range(33):
        if lm_id in landmarks_dict:
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

def create_skeleton_video(video_path, landmarks_frames, overlay_output, skeleton_output):
    """
    Creates both skeleton overlay and skeleton-only videos
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = None
    skeleton_writer = None
    
    if not os.path.exists(overlay_output):
        overlay_writer = cv2.VideoWriter(overlay_output, fourcc, fps, (width, height))
    
    if not os.path.exists(skeleton_output):
        skeleton_writer = cv2.VideoWriter(skeleton_output, fourcc, fps, (width, height))
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(landmarks_frames):
            landmarks_dict = landmarks_frames[frame_idx]
            
            # Create overlay video
            if overlay_writer is not None:
                overlay_frame = frame.copy()
                draw_skeleton(overlay_frame, landmarks_dict, width, height)
                overlay_writer.write(overlay_frame)
            
            # Create skeleton-only video
            if skeleton_writer is not None:
                skeleton_frame = np.zeros((height, width, 3), dtype=np.uint8)
                draw_skeleton(skeleton_frame, landmarks_dict, width, height)
                skeleton_writer.write(skeleton_frame)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)", end="\r")
    
    # Release resources
    cap.release()
    if overlay_writer:
        overlay_writer.release()
    if skeleton_writer:
        skeleton_writer.release()
    
    print(f"\nCreated videos: {os.path.basename(overlay_output)}, {os.path.basename(skeleton_output)}")

def create_3d_visualizations(video_id, strokes):
    """
    Create 3D visualizations of the normalized pose data
    """
    print(f"[STEP 3/3] Creating 3D visualizations for {len(strokes)} strokes from video_{video_id}")
    
    # Process each stroke
    successful = 0
    for stroke in strokes:
        stroke_dir = os.path.join(STROKES_LIBRARY, stroke)
        
        # Check if we have the normalized CSV
        norm_csv = os.path.join(stroke_dir, "stroke_norm.csv")
        html_output = os.path.join(stroke_dir, "stroke_3d_viz.html")
        
        if not os.path.exists(norm_csv):
            print(f"[SKIP] Missing normalized CSV for {stroke}")
            continue
        
        # If output already exists, skip
        if os.path.exists(html_output):
            print(f"[SKIP] 3D visualization already exists for {stroke}")
            successful += 1
            continue
        
        print(f"Creating 3D visualization for {stroke}...")
        
        try:
            # Load the CSV data
            df = pd.read_csv(norm_csv)
            
            # Extract landmark coordinates
            x_coords, y_coords, z_coords, vis_scores = extract_landmark_data(df)
            
            # Create the visualization
            fig = create_animated_pose_figure(x_coords, y_coords, z_coords, vis_scores)
            
            # Save the HTML file
            fig.write_html(html_output)
            successful += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to create 3D visualization for {stroke}: {e}")
    
    print(f"[DONE] Created 3D visualizations for {successful}/{len(strokes)} strokes")
    return successful > 0

def extract_landmark_data(df):
    """
    Extract landmark coordinates from DataFrame.
    """
    num_landmarks = 33
    x_coords = np.zeros((len(df), num_landmarks))
    y_coords = np.zeros((len(df), num_landmarks))
    z_coords = np.zeros((len(df), num_landmarks))
    vis_scores = np.zeros((len(df), num_landmarks))
    
    for i in range(num_landmarks):
        x_coords[:, i] = df[f'lm_{i}_x']
        y_coords[:, i] = df[f'lm_{i}_y']
        z_coords[:, i] = df[f'lm_{i}_z']
        if f'lm_{i}_vis' in df.columns:
            vis_scores[:, i] = df[f'lm_{i}_vis']
    
    return x_coords, y_coords, z_coords, vis_scores

def create_animated_pose_figure(x_coords, y_coords, z_coords, vis_scores):
    """
    Create an animated 3D plot of the pose sequence.
    """
    # Define MediaPipe Pose connections
    POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                       (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                       (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
                       (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                       (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for landmarks
    fig.add_trace(go.Scatter3d(
        x=x_coords[0],
        y=y_coords[0],
        z=z_coords[0],
        mode='markers',
        marker=dict(
            size=5,
            color=vis_scores[0],
            colorscale='Viridis',
            colorbar=dict(title='Visibility'),
            showscale=True
        ),
        name='Landmarks'
    ))
    
    # Add lines for connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        fig.add_trace(go.Scatter3d(
            x=[x_coords[0, start_idx], x_coords[0, end_idx]],
            y=[y_coords[0, start_idx], y_coords[0, end_idx]],
            z=[z_coords[0, start_idx], z_coords[0, end_idx]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # Create frames for animation
    frames = []
    for i in range(len(x_coords)):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x_coords[i],
                    y=y_coords[i],
                    z=z_coords[i],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=vis_scores[i],
                        colorscale='Viridis'
                    )
                )
            ] + [
                go.Scatter3d(
                    x=[x_coords[i, start_idx], x_coords[i, end_idx]],
                    y=[y_coords[i, start_idx], y_coords[i, end_idx]],
                    z=[z_coords[i, start_idx], z_coords[i, end_idx]],
                    mode='lines',
                    line=dict(color='red', width=2)
                )
                for start_idx, end_idx in POSE_CONNECTIONS
            ]
        )
        frames.append(frame)
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title='3D Pose Animation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]
            )]
        )],
        showlegend=True
    )
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Post-process a tennis video that's already in the pipeline")
    parser.add_argument("video_id", type=int, help="Video ID to process (e.g., 75 for video_75)")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip the stroke segmentation step")
    parser.add_argument("--skip-skeleton", action="store_true", help="Skip creating skeleton videos")
    parser.add_argument("--skip-3d", action="store_true", help="Skip creating 3D visualizations")
    
    args = parser.parse_args()
    
    video_id = args.video_id
    video_dir = os.path.join(VIDEOS_DIR, f"video_{video_id}")
    
    # Check if the video exists
    if not os.path.exists(video_dir):
        print(f"Error: video_{video_id} not found in {VIDEOS_DIR}")
        return 1
    
    # Step 1: Run stroke segmentation if not skipped
    if not args.skip_segmentation:
        run_stroke_segmentation()
    else:
        print("[SKIP] Stroke segmentation step skipped")
    
    # Get all strokes for this video
    strokes = get_strokes_for_video(video_id)
    if not strokes:
        print(f"No strokes found for video_{video_id} in the Strokes Library")
        print("Did you run the stroke segmentation step?")
        return 1
    
    print(f"Found {len(strokes)} strokes for video_{video_id}")
    
    # Step 2: Create skeleton videos if not skipped
    if not args.skip_skeleton:
        create_skeleton_videos(video_id, strokes)
    else:
        print("[SKIP] Skeleton video creation skipped")
    
    # Step 3: Create 3D visualizations if not skipped
    if not args.skip_3d:
        create_3d_visualizations(video_id, strokes)
    else:
        print("[SKIP] 3D visualization step skipped")
    
    print("\n=== PERSONAL POST-PROCESSING COMPLETE ===")
    print(f"Processed video_{video_id} with {len(strokes)} strokes")
    print(f"Results are available in the {STROKES_LIBRARY} directory")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 