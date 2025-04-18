#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_stroke_csv.py

Creates a visualization of stroke data from CSV file.
Draws both static plots and an animated skeleton.

Usage:
  python visualize_stroke_csv.py [stroke_id]

  - If stroke_id is not provided, defaults to 1497
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

# Define body connections 
POSE_CONNECTIONS_LEFT = [(11, 13), (13, 15), (23, 25), (25, 27)]
POSE_CONNECTIONS_RIGHT = [(12, 14), (14, 16), (24, 26), (26, 28)]
POSE_CONNECTIONS_CENTER = [(11, 12), (11, 23), (12, 24)]

# Define colors
COLOR_LEFT = 'blue'
COLOR_RIGHT = 'red'
COLOR_CENTER = 'green'

def load_stroke_data(csv_path):
    """Load stroke data from CSV file"""
    return pd.read_csv(csv_path)

def plot_static_trajectory(data, output_path):
    """Plot static trajectory of key points"""
    plt.figure(figsize=(12, 8))
    
    # Plot right wrist (lm_16) trajectory
    plt.subplot(2, 2, 1)
    plt.plot(data['lm_16_x'], data['lm_16_y'], 'r.-', label='Right Wrist')
    plt.gca().invert_yaxis()  # Flip y-axis to match image coordinates
    plt.title('Right Wrist Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    # Plot right elbow (lm_14) trajectory
    plt.subplot(2, 2, 2)
    plt.plot(data['lm_14_x'], data['lm_14_y'], 'b.-', label='Right Elbow')
    plt.gca().invert_yaxis()
    plt.title('Right Elbow Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    # Plot right elbow angle over time
    plt.subplot(2, 2, 3)
    plt.plot(data['frame_index'], data['right_elbow_angle'], 'g-')
    plt.title('Right Elbow Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    
    # Plot left elbow angle over time
    plt.subplot(2, 2, 4)
    plt.plot(data['frame_index'], data['left_elbow_angle'], 'm-')
    plt.title('Left Elbow Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Static plot saved to {output_path}")

def create_animation(data, output_path):
    """Create animation of the skeleton"""
    # Create figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.5, 0.5)
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Stroke Animation')
    
    # Initialize empty line objects for connections
    lines_left = [ax.plot([], [], 'b-', linewidth=2)[0] for _ in POSE_CONNECTIONS_LEFT]
    lines_right = [ax.plot([], [], 'r-', linewidth=2)[0] for _ in POSE_CONNECTIONS_RIGHT]
    lines_center = [ax.plot([], [], 'g-', linewidth=2)[0] for _ in POSE_CONNECTIONS_CENTER]
    
    # Initialize empty scatter for keypoints
    keypoints = ax.scatter([], [], s=50, c='k', zorder=3)
    
    # Initialize frame counter text
    frame_text = ax.text(0.02, 0.02, '', transform=ax.transAxes)
    
    # Initialize right elbow angle text
    angle_text = ax.text(0.02, 0.06, '', transform=ax.transAxes)
    
    def init():
        """Initialize animation"""
        for line in lines_left + lines_right + lines_center:
            line.set_data([], [])
        keypoints.set_offsets(np.empty((0, 2)))
        frame_text.set_text('')
        angle_text.set_text('')
        return lines_left + lines_right + lines_center + [keypoints, frame_text, angle_text]
    
    def update(frame):
        """Update animation for each frame"""
        # Clear previous frame
        for line in lines_left + lines_right + lines_center:
            line.set_data([], [])
        
        # Extract keypoint positions for this frame
        frame_data = data.iloc[frame]
        
        # Update left connections
        for i, (a, b) in enumerate(POSE_CONNECTIONS_LEFT):
            x_a, y_a = frame_data[f'lm_{a}_x'], frame_data[f'lm_{a}_y']
            x_b, y_b = frame_data[f'lm_{b}_x'], frame_data[f'lm_{b}_y']
            lines_left[i].set_data([x_a, x_b], [y_a, y_b])
        
        # Update right connections
        for i, (a, b) in enumerate(POSE_CONNECTIONS_RIGHT):
            x_a, y_a = frame_data[f'lm_{a}_x'], frame_data[f'lm_{a}_y']
            x_b, y_b = frame_data[f'lm_{b}_x'], frame_data[f'lm_{b}_y']
            lines_right[i].set_data([x_a, x_b], [y_a, y_b])
        
        # Update center connections
        for i, (a, b) in enumerate(POSE_CONNECTIONS_CENTER):
            x_a, y_a = frame_data[f'lm_{a}_x'], frame_data[f'lm_{a}_y']
            x_b, y_b = frame_data[f'lm_{b}_x'], frame_data[f'lm_{b}_y']
            lines_center[i].set_data([x_a, x_b], [y_a, y_b])
        
        # Collect all keypoints for this frame
        keypoint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        xy_data = np.array([[frame_data[f'lm_{i}_x'], frame_data[f'lm_{i}_y']] for i in keypoint_indices])
        keypoints.set_offsets(xy_data)
        
        # Update frame counter
        frame_text.set_text(f'Frame: {frame_data["frame_index"]}/{len(data)-1}')
        
        # Update angle text
        angle_text.set_text(f'Right Elbow: {frame_data["right_elbow_angle"]:.1f}°  Left Elbow: {frame_data["left_elbow_angle"]:.1f}°')
        
        return lines_left + lines_right + lines_center + [keypoints, frame_text, angle_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=range(len(data)),
        init_func=init, blit=True, interval=50
    )
    
    # Save animation
    anim.save(output_path, writer='pillow', fps=15)
    print(f"Animation saved to {output_path}")
    
    plt.close(fig)

def main():
    # Parse command line arguments
    stroke_id = "1497"  # default
    
    if len(sys.argv) > 1:
        stroke_id = sys.argv[1]
    
    # Define paths
    stroke_dir = f"Strokes_Library/stroke_{stroke_id}"
    csv_path = f"{stroke_dir}/stroke.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CSV exists
    if not os.path.isfile(csv_path):
        print(f"[ERROR] Stroke CSV not found: {csv_path}")
        print(f"Make sure the stroke ID is correct and the file exists.")
        return
    
    # Load data
    print(f"[INFO] Loading data from {csv_path}")
    data = load_stroke_data(csv_path)
    print(f"[INFO] Loaded {len(data)} frames of data")
    
    # Create static trajectory plot
    plot_output = f"{output_dir}/stroke_{stroke_id}_trajectory.png"
    plot_static_trajectory(data, plot_output)
    
    # Create animation
    animation_output = f"{output_dir}/stroke_{stroke_id}_animation.gif"
    create_animation(data, animation_output)

if __name__ == "__main__":
    main() 