#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_video_skeleton.py

Takes a raw video file, runs MediaPipe Pose detection on it,
and creates skeleton visualization outputs (skeleton-only and overlay videos).

Usage:
  python process_video_skeleton.py [video_path] [output_prefix]

  - video_path is required
  - output_prefix is optional (defaults to the video filename without extension)

DEPENDENCIES:
    pip install opencv-python mediapipe numpy
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import argparse
import time

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

def draw_pretty_skeleton(frame_bgr, landmarks, width, height):
    """
    Draws a color-coded skeleton with thicker lines and anti-aliasing.
    """
    # Center lines
    for (lmA, lmB) in POSE_CONNECTIONS_CENTER:
        if lmA in landmarks and lmB in landmarks:
            ptA = (int(landmarks[lmA][0] * width), int(landmarks[lmA][1] * height))
            ptB = (int(landmarks[lmB][0] * width), int(landmarks[lmB][1] * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_CENTER, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Left edges
    for (lmA, lmB) in POSE_CONNECTIONS_LEFT:
        if lmA in landmarks and lmB in landmarks:
            ptA = (int(landmarks[lmA][0] * width), int(landmarks[lmA][1] * height))
            ptB = (int(landmarks[lmB][0] * width), int(landmarks[lmB][1] * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_LEFT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Right edges
    for (lmA, lmB) in POSE_CONNECTIONS_RIGHT:
        if lmA in landmarks and lmB in landmarks:
            ptA = (int(landmarks[lmA][0] * width), int(landmarks[lmA][1] * height))
            ptB = (int(landmarks[lmB][0] * width), int(landmarks[lmB][1] * height))
            cv2.line(frame_bgr, ptA, ptB, COLOR_RIGHT, thickness=LINE_THICKNESS, lineType=LINE_TYPE)

    # Keypoints
    for lm_id, (nx, ny) in landmarks.items():
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
    
    return frame_bgr

def process_video(video_path, output_prefix):
    """
    Process a video with MediaPipe Pose and create output videos
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video: {width}x{height} at {fps} FPS, {total_frames} frames")
    
    # Create output directory if it doesn't exist
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup output video writers
    skeleton_output = f"{output_dir}/{output_prefix}_skeleton.mp4"
    overlay_output = f"{output_dir}/{output_prefix}_overlay.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_writer = cv2.VideoWriter(skeleton_output, fourcc, fps, (width, height))
    overlay_writer = cv2.VideoWriter(overlay_output, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing video and creating:")
    print(f"       - {skeleton_output}")
    print(f"       - {overlay_output}")
    
    # Process frame by frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # If pose detected, extract landmarks
        if results.pose_landmarks:
            # Convert landmarks to dictionary format {lm_id: (x, y)}
            landmarks = {}
            for lm_id, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[lm_id] = (landmark.x, landmark.y)
            
            # Create black background for skeleton-only video
            skeleton_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw skeleton on black background
            draw_pretty_skeleton(skeleton_frame, landmarks, width, height)
            
            # Draw skeleton on original frame (for overlay)
            overlay_frame = frame.copy()
            draw_pretty_skeleton(overlay_frame, landmarks, width, height)
            
            # Write frames to output videos
            skeleton_writer.write(skeleton_frame)
            overlay_writer.write(overlay_frame)
        else:
            # No pose detected, just write original frame to overlay 
            # and black frame to skeleton
            skeleton_writer.write(np.zeros((height, width, 3), dtype=np.uint8))
            overlay_writer.write(frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            percent = (frame_count / total_frames) * 100
            print(f"[INFO] Progress: {frame_count}/{total_frames} frames ({percent:.1f}%)")
    
    # Release resources
    cap.release()
    skeleton_writer.release()
    overlay_writer.release()
    pose.close()
    
    print(f"[DONE] Successfully processed video")
    return skeleton_output, overlay_output

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python process_video_skeleton.py [video_path] [output_prefix]")
        print("  - video_path: Path to input video file")
        print("  - output_prefix: Optional name prefix for output files")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    # Use video filename (without extension) as default output prefix
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]
    else:
        output_prefix = os.path.splitext(os.path.basename(video_path))[0]
    
    # Process the video
    skeleton_output, overlay_output = process_video(video_path, output_prefix)

if __name__ == "__main__":
    main() 