import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Union

# Define paths
VIDEO_PATH = "test_videos/test_video.mp4"
CSV_PATH = "test_videos/test_landmarks.csv"
OUTPUT_DIR = "output"

# Drawing options
CONNECTIONS = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
               (0, 7), (7, 8), (8, 9), (9, 10), (7, 11), (11, 12), (12, 13),
               (7, 14), (14, 15), (15, 16)]
COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

def convert_video_to_mp4(input_path: str) -> str:
    """
    Convert video to MP4 format if it's not already in MP4 format
    
    Args:
        input_path (str): Path to the input video file
        
    Returns:
        str: Path to the MP4 video file
    """
    # If the file is already MP4, return the path
    if input_path.lower().endswith(".mp4"):
        return input_path
        
    # Create output path with .mp4 extension
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}.mp4"
    
    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"MP4 version already exists at {output_path}")
        return output_path
    
    print(f"Converting {input_path} to MP4 format...")
    # Convert the video using ffmpeg
    try:
        subprocess.run([
            "ffmpeg", "-i", input_path, 
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ], check=True)
        print(f"Conversion complete: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return input_path  # Return original path if conversion fails

def load_landmarks_from_csv(csv_path: str) -> Dict[int, List[List[float]]]:
    """
    Load landmarks from a CSV file
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        Dict[int, List[List[float]]]: Dictionary of landmarks for each frame
    """
    data = pd.read_csv(csv_path)
    landmarks_by_frame = {}
    
    for _, row in data.iterrows():
        frame_idx = int(row['frame_idx'])
        landmarks = []
        
        for i in range(33):  # 33 landmarks in MediaPipe Pose
            x = row[f'x_{i}']
            y = row[f'y_{i}']
            z = row[f'z_{i}']
            landmarks.append([x, y, z])
        
        landmarks_by_frame[frame_idx] = landmarks
    
    return landmarks_by_frame

def draw_landmarks_on_frame(
    frame: np.ndarray, 
    landmarks: List[List[float]],
    connections: List[Tuple[int, int]] = CONNECTIONS,
    colors: List[Tuple[int, int, int]] = COLORS
) -> np.ndarray:
    """
    Draw landmarks and connections on a frame
    
    Args:
        frame (np.ndarray): Input frame
        landmarks (List[List[float]]): List of landmarks
        connections (List[Tuple[int, int]]): List of connections between landmarks
        colors (List[Tuple[int, int, int]]): List of colors
        
    Returns:
        np.ndarray: Frame with landmarks and connections
    """
    h, w, _ = frame.shape
    
    # Draw landmarks
    for landmark in landmarks:
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # Draw connections
    for i, connection in enumerate(connections):
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start[0] * w), int(start[1] * h))
            end_point = (int(end[0] * w), int(end[1] * h))
            
            color = colors[i % len(colors)]
            cv2.line(frame, start_point, end_point, color, 2)
    
    return frame

def process_mp4(
    video_path: str, 
    csv_path: Optional[str] = None, 
    output_dir: str = OUTPUT_DIR, 
    connections: List[Tuple[int, int]] = CONNECTIONS,
    colors: List[Tuple[int, int, int]] = COLORS
) -> Tuple[str, str, str]:
    """
    Process an MP4 file and create three videos:
    1. Raw video
    2. Overlay video with skeleton
    3. Skeleton video (black background)
    
    Args:
        video_path (str): Path to the input video
        csv_path (Optional[str]): Path to the CSV file with landmarks
        output_dir (str): Directory to save the output videos
        connections (List[Tuple[int, int]]): List of connections between landmarks
        colors (List[Tuple[int, int, int]]): List of colors
        
    Returns:
        Tuple[str, str, str]: Paths to raw, overlay and skeleton videos
    """
    # Ensure video is in MP4 format
    video_path = convert_video_to_mp4(video_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output paths
    raw_path = os.path.join(output_dir, f"{video_name}_raw.mp4")
    overlay_path = os.path.join(output_dir, f"{video_name}_overlay.mp4")
    skeleton_path = os.path.join(output_dir, f"{video_name}_skeleton.mp4")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
    overlay_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))
    skeleton_writer = cv2.VideoWriter(skeleton_path, fourcc, fps, (width, height))
    
    # Load landmarks from CSV if provided
    landmarks_by_frame = {}
    if csv_path:
        landmarks_by_frame = load_landmarks_from_csv(csv_path)
    
    # MediaPipe pose detector (as fallback if CSV not provided)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                        smooth_landmarks=True, min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Copy frame for raw video
        raw_frame = frame.copy()
        
        # Create overlay video frame
        overlay_frame = frame.copy()
        
        # Create skeleton video frame (black background)
        skeleton_frame = np.zeros_like(frame)
        
        # Get landmarks for the current frame
        if frame_idx in landmarks_by_frame:
            # Use landmarks from CSV
            landmarks = landmarks_by_frame[frame_idx]
        else:
            # Use MediaPipe to detect landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
            else:
                landmarks = []
        
        # Draw landmarks on frames if available
        if landmarks:
            overlay_frame = draw_landmarks_on_frame(overlay_frame, landmarks, connections, colors)
            skeleton_frame = draw_landmarks_on_frame(skeleton_frame, landmarks, connections, colors)
        
        # Write frames to output videos
        raw_writer.write(raw_frame)
        overlay_writer.write(overlay_frame)
        skeleton_writer.write(skeleton_frame)
        
        frame_idx += 1
        
        # Print progress
        if frame_idx % 30 == 0:
            progress = (frame_idx / frame_count) * 100
            print(f"Progress: {progress:.2f}% ({frame_idx}/{frame_count})")
    
    # Release resources
    cap.release()
    raw_writer.release()
    overlay_writer.release()
    skeleton_writer.release()
    
    return raw_path, overlay_path, skeleton_path

if __name__ == "__main__":
    raw, overlay, skeleton = process_mp4(VIDEO_PATH, CSV_PATH)
    print(f"Raw video: {raw}")
    print(f"Overlay video: {overlay}")
    print(f"Skeleton video: {skeleton}") 