import os
import csv
import re
import math
import numpy as np
import cv2
import ffmpeg
from scipy.signal import savgol_filter, find_peaks

#############################################
# CONFIGURATION
#############################################
# Path settings
NORMALIZED_CSV = "tennis_clips/normalized/video_1_normalized.csv"  # your normalized CSV file
ORIGINAL_VIDEO = "tennis_clips/videos/video_1.mp4"                 # original video corresponding to CSV
OUTPUT_STROKE_FOLDER = "tennis_clips/strokes"                       # where to save stroke clips
os.makedirs(OUTPUT_STROKE_FOLDER, exist_ok=True)

# Video/processing parameters
FPS = 30                           # frame rate of your video

# For composite metric, we choose landmarks:
# Left shoulder: 11, right shoulder: 12,
# Left elbow: 13, right elbow: 14,
# Left wrist: 15, right wrist: 16,
# Left hip: 23, right hip: 24
SELECTED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]

# Smoothing parameters for the acceleration signal
SMOOTH_WINDOW = 61                # must be odd
SMOOTH_POLY_ORDER = 2

# Peak detection parameters (on the inverted composite acceleration signal)
ACCEL_PROMINENCE = 0.1            # adjust this to ignore small “valleys”

# Minimum stroke duration (in seconds)
MIN_STROKE_DURATION_SEC = 3

# Optional: Instead of a fixed margin, we use the average between consecutive stroke boundaries.
# (Set VIDEO_MARGIN_SEC to 0 to use exact midpoints.)
VIDEO_MARGIN_SEC = 0

#############################################
# FUNCTIONS FOR COMPOSITE METRIC
#############################################
def load_normalized_csv(csv_path):
    """
    Load the normalized CSV.
    We assume the first column is frame_index, then 33*4 columns of raw pose data,
    then additional columns. We only use the first (33*4 + 1) columns.
    """
    data = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # We assume raw data columns: 1 + 33*4 = 133 columns (frame_index plus 132 columns)
        raw_columns = 1 + 33 * 4
        for row in reader:
            # Convert first raw_columns to float
            data.append([float(x) for x in row[:raw_columns]])
    return data

def extract_landmark_xy(frame_row, lm_index):
    """
    Given a frame row (list of floats) from the CSV,
    extract normalized x and y for landmark lm_index.
    The CSV columns are assumed to be:
      frame_index, lm_0_x, lm_0_y, lm_0_z, lm_0_vis, lm_1_x, ...
    so for landmark i, x is at 1 + i*4 and y is at 1 + i*4 + 1.
    """
    base = 1 + lm_index * 4
    x = frame_row[base]
    y = frame_row[base + 1]
    return x, y

def compute_composite_velocity(data):
    """
    Compute a composite velocity signal for each frame based on the selected landmarks.
    For each frame (except the first), compute the Euclidean distance
    (in the normalized 2D plane) of each selected landmark from its previous frame,
    then average them.
    Returns an array of velocity values (length = num_frames - 1).
    """
    velocities = []
    for i in range(1, len(data)):
        diffs = []
        for lm in SELECTED_LANDMARKS:
            x1, y1 = extract_landmark_xy(data[i-1], lm)
            x2, y2 = extract_landmark_xy(data[i], lm)
            diff = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            diffs.append(diff)
        velocities.append(np.mean(diffs))
    return np.array(velocities)

def compute_composite_acceleration(velocities, fps):
    """
    Compute composite acceleration from velocity signal using numerical differentiation.
    Multiply by fps to scale per second.
    Returns an array of acceleration values (length = len(velocities) - 1).
    """
    # Use np.diff to compute difference between successive velocity values
    acceleration = np.diff(velocities) * fps
    return acceleration

def smooth_signal(signal, window, poly_order):
    """
    Apply Savitzky–Golay smoothing filter.
    """
    # If signal length is shorter than window, adjust window
    if len(signal) < window:
        window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    return savgol_filter(signal, window_length=window, polyorder=poly_order)

def detect_stroke_boundaries(accel, prominence):
    """
    Given the composite acceleration signal, detect stroke boundaries.
    We assume that a stroke boundary corresponds to a local minimum in the acceleration.
    Since find_peaks finds local maxima, we invert the signal.
    Returns the indices (in the acceleration signal) of the valleys.
    """
    inverted = -accel  # invert so that valleys become peaks
    peaks, properties = find_peaks(inverted, prominence=prominence)
    return peaks

#############################################
# VIDEO CLIPPING FUNCTIONS
#############################################
def clip_video(video_path, stroke_boundaries, fps, margin_sec=0):
    """
    Given the original video file and a list of stroke boundary frame indices,
    clip the video into stroke segments. We define a stroke segment as the video between
    two consecutive boundaries. The clip time (in seconds) for a stroke is taken as the midpoint
    between the two boundaries, optionally extended by margin_sec.
    """
    if len(stroke_boundaries) < 2:
        print("Not enough boundaries detected to clip strokes.")
        return

    # Convert frame indices to seconds
    times = [idx / fps for idx in stroke_boundaries]
    # Use midpoints between consecutive boundaries as stroke cut points
    stroke_cut_times = []
    for i in range(len(times)-1):
        stroke_cut_times.append((times[i] + times[i+1]) / 2)
    
    # Now, define stroke segments as [t_start, t_end] between successive cut times.
    # Also enforce a minimum stroke duration.
    stroke_segments = []
    for i in range(len(stroke_cut_times)-1):
        t_start = stroke_cut_times[i] - margin_sec
        t_end = stroke_cut_times[i+1] + margin_sec
        duration = t_end - t_start
        if duration < MIN_STROKE_DURATION_SEC:
            print(f"Skipping stroke segment {i+1} (duration {duration:.2f}s too short)")
            continue
        stroke_segments.append((max(0, t_start), t_end))
    
    # Clip the video segments using ffmpeg
    for i, (t_start, t_end) in enumerate(stroke_segments):
        output_clip = os.path.join(OUTPUT_STROKE_FOLDER, f"stroke_{i+1}.mp4")
        print(f"Clipping stroke {i+1}: start={t_start:.2f}s, end={t_end:.2f}s, duration={t_end - t_start:.2f}s")
        try:
            (
                ffmpeg
                .input(video_path, ss=t_start, to=t_end)
                .output(output_clip, codec="copy", loglevel="error", y=None)
                .run()
            )
        except Exception as e:
            print(f"Error clipping stroke {i+1}: {e}")
    
    print("Stroke clipping complete.")

#############################################
# MAIN SCRIPT
#############################################
def main():
    # 1. Load normalized CSV data
    print("Loading normalized CSV data...")
    data = load_normalized_csv(NORMALIZED_CSV)
    if not data:
        print("No data loaded from CSV!")
        return

    # 2. Compute composite velocity and acceleration
    print("Computing composite velocity...")
    comp_velocity = compute_composite_velocity(data)
    print("Computing composite acceleration...")
    comp_acceleration = compute_composite_acceleration(comp_velocity, FPS)
    
    # 3. Smooth the acceleration signal
    print("Smoothing acceleration signal...")
    smooth_accel = smooth_signal(comp_acceleration, SMOOTH_WINDOW, SMOOTH_POLY_ORDER)
    
    # 4. Detect stroke boundaries (valleys)
    print("Detecting stroke boundaries...")
    boundary_indices = detect_stroke_boundaries(smooth_accel, ACCEL_PROMINENCE)
    # Note: boundary_indices are relative to the acceleration array, which is one less than velocity frames.
    # To map back to the original frame indices, we add 1 (or choose a suitable offset).
    stroke_boundaries = (boundary_indices + 1).tolist()
    print(f"Detected {len(stroke_boundaries)} stroke boundaries at frames: {stroke_boundaries}")
    
    # 5. For stroke segmentation, we want to pair consecutive boundaries and take midpoints.
    # Optionally, filter out segments that are too short.
    # (We then clip the video accordingly.)
    print("Clipping strokes from video...")
    clip_video(ORIGINAL_VIDEO, stroke_boundaries, FPS, margin_sec=VIDEO_MARGIN_SEC)

if __name__ == '__main__':
    main()
