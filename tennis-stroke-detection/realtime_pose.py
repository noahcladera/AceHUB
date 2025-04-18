#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
realtime_pose.py

Simple script to run MediaPipe Pose on a video file or webcam input
and display the results in real-time.

Usage:
    python realtime_pose.py --video path/to/video.mp4  # Process a video file
    python realtime_pose.py  # Use webcam (default)

Controls:
    Press 'q' or 'ESC' to quit
    Press 's' to save a screenshot

Dependencies:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MediaPipe Pose on a video or webcam')
    parser.add_argument('--video', type=str, default='', help='Path to video file (empty for webcam)')
    args = parser.parse_args()

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Setup video source (file or webcam)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Processing video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Using webcam input")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")

    # Performance metrics
    start_time = time.time()
    frame_count = 0

    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Convert to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            # Standard MediaPipe visualization
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add FPS counter
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
                start_time = time.time()
                frame_count = 0
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('MediaPipe Pose', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break
        elif key == ord('s'):  # 's' to save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pose_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main() 