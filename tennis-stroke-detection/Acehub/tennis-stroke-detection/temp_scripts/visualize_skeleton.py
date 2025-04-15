#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import argparse

# Define the full set of MediaPipe pose connections
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                   (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                   (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
                   (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                   (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]

# These are the core connections that should show a basic human form
CORE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (23, 25), (25, 27),  # left leg
    (12, 24), (24, 26), (26, 28),  # right leg
    (23, 24),  # hips
]

def load_landmarks_from_csv(csv_path):
    """Load landmark data from CSV file"""
    df = pd.read_csv(csv_path)
    frames = []
    
    for _, row in df.iterrows():
        landmarks = {}
        for i in range(33):  # MediaPipe has 33 landmarks
            if f'lm_{i}_x' in df.columns and f'lm_{i}_y' in df.columns:
                x = row[f'lm_{i}_x']
                y = row[f'lm_{i}_y']
                landmarks[i] = (x, y)
        frames.append(landmarks)
    
    return frames

def plot_static_frames(frames, output_dir='skeleton_frames'):
    """Generate static plots of selected frames"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select frames to plot (first, middle, last)
    total_frames = len(frames)
    frames_to_plot = [0, total_frames // 2, total_frames - 1]
    
    for frame_idx in frames_to_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        landmarks = frames[frame_idx]
        
        # Plot connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in landmarks and end_idx in landmarks:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 'r-', linewidth=2)
        
        # Plot points
        x_coords = [landmarks[i][0] for i in landmarks]
        y_coords = [landmarks[i][1] for i in landmarks]
        ax.scatter(x_coords, y_coords, c='blue', s=30)
        
        # Set plot properties
        ax.set_xlim(-2, 2)
        ax.set_ylim(2, -2)  # Invert Y-axis to match image coordinates
        ax.set_title(f'Frame {frame_idx}')
        ax.grid(True)
        
        plt.savefig(f'{output_dir}/frame_{frame_idx}.png')
        plt.close()
        
    print(f"Static frames saved to {output_dir}")

def create_animation(frames, output_path='skeleton_animation.mp4'):
    """Create an animation of the skeleton movement"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set plot limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(2, -2)  # Invert Y-axis to match image coordinates
    ax.set_title('Skeleton Animation')
    ax.grid(True)
    
    # Create a line for each connection
    lines = []
    for _ in POSE_CONNECTIONS:
        line, = ax.plot([], [], 'r-', linewidth=2)
        lines.append(line)
    
    # Create scatter for landmarks
    scatter = ax.scatter([], [], c='blue', s=30)
    
    def init():
        for line in lines:
            line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return lines + [scatter]
    
    def update(frame_idx):
        ax.set_title(f'Frame {frame_idx}')
        landmarks = frames[frame_idx]
        
        # Update connections
        for i, connection in enumerate(POSE_CONNECTIONS):
            start_idx, end_idx = connection
            if start_idx in landmarks and end_idx in landmarks:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                lines[i].set_data([start_point[0], end_point[0]], 
                                [start_point[1], end_point[1]])
            else:
                lines[i].set_data([], [])
        
        # Update points
        valid_landmarks = [landmarks[i] for i in landmarks]
        if valid_landmarks:
            scatter.set_offsets(valid_landmarks)
        
        return lines + [scatter]
    
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(frames), 
        init_func=init, 
        blit=True,
        interval=50  # 50 ms between frames
    )
    
    # Save animation
    ani.save(output_path, writer='ffmpeg', fps=20)
    print(f"Animation saved to {output_path}")
    
    plt.close()

def visualize_with_core_connections(frames, output_path='core_skeleton_animation.mp4'):
    """Create an animation showing only the core body connections"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set plot limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(2, -2)  # Invert Y-axis to match image coordinates
    ax.set_title('Core Skeleton Animation')
    ax.grid(True)
    
    # Create a line for each connection
    lines = []
    for _ in CORE_CONNECTIONS:
        line, = ax.plot([], [], 'b-', linewidth=3)
        lines.append(line)
    
    # Create scatter for landmarks
    scatter = ax.scatter([], [], c='red', s=50)
    
    def init():
        for line in lines:
            line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return lines + [scatter]
    
    def update(frame_idx):
        ax.set_title(f'Frame {frame_idx}')
        landmarks = frames[frame_idx]
        
        # Update connections
        for i, connection in enumerate(CORE_CONNECTIONS):
            start_idx, end_idx = connection
            if start_idx in landmarks and end_idx in landmarks:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                lines[i].set_data([start_point[0], end_point[0]], 
                                [start_point[1], end_point[1]])
            else:
                lines[i].set_data([], [])
        
        # Update points - only include core points
        core_indices = set()
        for start_idx, end_idx in CORE_CONNECTIONS:
            core_indices.add(start_idx)
            core_indices.add(end_idx)
            
        valid_landmarks = [landmarks[i] for i in core_indices if i in landmarks]
        if valid_landmarks:
            scatter.set_offsets(valid_landmarks)
        
        return lines + [scatter]
    
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(frames), 
        init_func=init, 
        blit=True,
        interval=50  # 50 ms between frames
    )
    
    # Save animation
    ani.save(output_path, writer='ffmpeg', fps=20)
    print(f"Core animation saved to {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize skeleton data from CSV file')
    parser.add_argument('csv_file', help='Path to the normalized CSV file with landmark data')
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv_file}")
    frames = load_landmarks_from_csv(args.csv_file)
    print(f"Loaded {len(frames)} frames")
    
    # Create output directory based on input filename
    base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
    output_dir = f"visualization_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    plot_static_frames(frames, output_dir=f"{output_dir}/frames")
    create_animation(frames, output_path=f"{output_dir}/full_skeleton.mp4")
    visualize_with_core_connections(frames, output_path=f"{output_dir}/core_skeleton.mp4")
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 