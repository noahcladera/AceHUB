def create_overlay_and_skeleton_videos(video_path, overlay_video_path, skeleton_video_path, csv_path=None):
    """Create overlay and skeleton videos based on the source video and landmarks.
    
    Args:
        video_path: Path to the source video
        overlay_video_path: Path to save the overlay video
        skeleton_video_path: Path to save the skeleton video
        csv_path: Optional path to the landmarks CSV file
    """
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import mediapipe as mp
    from typing import List, Tuple
    
    # Define skeleton connections (MediaPipe pose model)
    CONNECTIONS = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
                  (0, 7), (7, 8), (8, 9), (9, 10), (7, 11), (11, 12), (12, 13),
                  (7, 14), (14, 15), (15, 16)]
    COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    
    # Create the output directories if they don't exist
    os.makedirs(os.path.dirname(overlay_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(skeleton_video_path), exist_ok=True)
    
    # Check if the video exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
        
    # Load landmarks from CSV if provided
    landmarks_by_frame = {}
    if csv_path and os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        for _, row in data.iterrows():
            frame_idx = int(row['frame_idx'])
            landmarks = []
            
            for i in range(33):  # 33 landmarks in MediaPipe Pose
                x = row[f'x_{i}']
                y = row[f'y_{i}']
                z = row[f'z_{i}']
                landmarks.append([x, y, z])
            
            landmarks_by_frame[frame_idx] = landmarks
    
    # MediaPipe pose detector (as fallback if CSV not provided or for frames missing in CSV)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                       smooth_landmarks=True, min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
    skeleton_writer = cv2.VideoWriter(skeleton_video_path, fourcc, fps, (width, height))
    
    def draw_landmarks_on_frame(
        frame: np.ndarray, 
        landmarks: List[List[float]],
        connections: List[Tuple[int, int]] = CONNECTIONS,
        colors: List[Tuple[int, int, int]] = COLORS
    ) -> np.ndarray:
        """Draw landmarks and connections on a frame"""
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
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
            overlay_frame = draw_landmarks_on_frame(overlay_frame, landmarks, CONNECTIONS, COLORS)
            skeleton_frame = draw_landmarks_on_frame(skeleton_frame, landmarks, CONNECTIONS, COLORS)
        
        # Write frames to output videos
        overlay_writer.write(overlay_frame)
        skeleton_writer.write(skeleton_frame)
        
        frame_idx += 1
        
        # Print progress occasionally
        if frame_idx % 30 == 0:
            progress = (frame_idx / frame_count) * 100
            print(f"Progress: {progress:.2f}% ({frame_idx}/{frame_count})")
    
    # Release resources
    cap.release()
    overlay_writer.release()
    skeleton_writer.release()
    
    print(f"Created overlay video: {overlay_video_path}")
    print(f"Created skeleton video: {skeleton_video_path}") 