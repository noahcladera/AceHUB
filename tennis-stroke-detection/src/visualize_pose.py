#!/usr/bin/env python3
"""
3D Pose Visualization Script

This script creates an interactive 3D visualization of pose data from CSV files.
The CSV files should contain MediaPipe Pose landmark coordinates (x, y, z) and visibility scores.

Dependencies:
    - numpy
    - pandas
    - plotly
    - mediapipe (for pose connections)

Usage:
    python visualize_pose.py [csv_file_path]

If no CSV file path is provided, it will use a default path.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mediapipe as mp

# MediaPipe Pose connections
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def load_pose_data(csv_path):
    """
    Load pose data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing pose data
        
    Returns:
        pandas.DataFrame: DataFrame containing the pose data
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def extract_landmarks(df):
    """
    Extract landmark coordinates from DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing pose data
        
    Returns:
        tuple: (x_coords, y_coords, z_coords, vis_scores)
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

def create_3d_pose_figure(x_coords, y_coords, z_coords, vis_scores, frame_idx=0):
    """
    Create a 3D plot of the pose for a specific frame.
    
    Args:
        x_coords (numpy.ndarray): X coordinates for all landmarks
        y_coords (numpy.ndarray): Y coordinates for all landmarks
        z_coords (numpy.ndarray): Z coordinates for all landmarks
        vis_scores (numpy.ndarray): Visibility scores for all landmarks
        frame_idx (int): Index of the frame to plot
        
    Returns:
        plotly.graph_objects.Figure: 3D plot of the pose
    """
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for landmarks
    fig.add_trace(go.Scatter3d(
        x=x_coords[frame_idx],
        y=y_coords[frame_idx],
        z=z_coords[frame_idx],
        mode='markers',
        marker=dict(
            size=5,
            color=vis_scores[frame_idx],
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
            x=[x_coords[frame_idx, start_idx], x_coords[frame_idx, end_idx]],
            y=[y_coords[frame_idx, start_idx], y_coords[frame_idx, end_idx]],
            z=[z_coords[frame_idx, start_idx], z_coords[frame_idx, end_idx]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='3D Pose Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    return fig

def create_animated_pose_figure(x_coords, y_coords, z_coords, vis_scores):
    """
    Create an animated 3D plot of the pose sequence.
    
    Args:
        x_coords (numpy.ndarray): X coordinates for all landmarks
        y_coords (numpy.ndarray): Y coordinates for all landmarks
        z_coords (numpy.ndarray): Z coordinates for all landmarks
        vis_scores (numpy.ndarray): Visibility scores for all landmarks
        
    Returns:
        plotly.graph_objects.Figure: Animated 3D plot of the pose sequence
    """
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
    # Default CSV path (can be changed or passed as command-line argument)
    default_csv_path = "videos/video_1/video_1_normalized.csv"
    
    # Get CSV path from command line or use default
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv_path
    
    try:
        # Load pose data
        df = load_pose_data(csv_path)
        
        # Extract landmark coordinates
        x_coords, y_coords, z_coords, vis_scores = extract_landmarks(df)
        
        # Create static visualization
        static_fig = create_3d_pose_figure(x_coords, y_coords, z_coords, vis_scores)
        static_fig.show()
        
        # Create animated visualization
        animated_fig = create_animated_pose_figure(x_coords, y_coords, z_coords, vis_scores)
        animated_fig.show()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 