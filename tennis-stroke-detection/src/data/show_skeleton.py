import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CSV_PATH = ""

# A minimal set of edges to draw a rough skeleton (MediaPipe Pose style).
# Add or remove edges as desired.
POSE_CONNECTIONS = [
    (11, 13),  # left shoulder -> left elbow
    (13, 15),  # left elbow   -> left wrist
    (12, 14),  # right shoulder -> right elbow
    (14, 16),  # right elbow    -> right wrist
    (11, 12),  # left shoulder -> right shoulder
    (11, 23),  # left shoulder -> left hip
    (12, 24),  # right shoulder -> right hip
    (23, 25),  # left hip -> left knee
    (25, 27),  # left knee -> left ankle
    (24, 26),  # right hip -> right knee
    (26, 28),  # right knee -> right ankle
]

def load_pose_frames(csv_path):
    """
    Reads a CSV with columns like: frame_index, lm_0_x, lm_0_y, ...
    Returns a list of dict, where frames[i][lm_id] = (x, y).
    """
    frames = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Identify which columns hold x,y for each of the 33 landmarks
        landmark_xy = {}
        for lm_id in range(33):
            x_key = f"lm_{lm_id}_x"
            y_key = f"lm_{lm_id}_y"
            if x_key in header and y_key in header:
                x_idx = header.index(x_key)
                y_idx = header.index(y_key)
                landmark_xy[lm_id] = (x_idx, y_idx)

        for row in reader:
            lm_dict = {}
            for lm_id in range(33):
                if lm_id in landmark_xy:
                    x_idx, y_idx = landmark_xy[lm_id]
                    x_val = float(row[x_idx])
                    y_val = float(row[y_idx])
                    lm_dict[lm_id] = (x_val, y_val)
                else:
                    lm_dict[lm_id] = (0.0, 0.0)
            frames.append(lm_dict)
    return frames

def init_func():
    """
    Called once at the start of FuncAnimation.
    We won't draw anything yet, just return the lines/points that we’ll update.
    """
    for line in skeleton_lines:
        line.set_data([], [])
    scatter_plot.set_offsets(np.array([]).reshape(0, 2))
    return skeleton_lines + [scatter_plot]

def update_func(frame_idx):
    """
    Called for each frame to update the skeleton lines and keypoints.
    """
    landmarks = frames[frame_idx]

    # Update each skeleton edge
    for i, (lmA, lmB) in enumerate(POSE_CONNECTIONS):
        xA, yA = landmarks[lmA]
        xB, yB = landmarks[lmB]
        skeleton_lines[i].set_data([xA, xB], [yA, yB])

    # Update the scatter of all points
    xs = [landmarks[lm_id][0] for lm_id in range(33)]
    ys = [landmarks[lm_id][1] for lm_id in range(33)]
    scatter_plot.set_offsets(np.column_stack([xs, ys]))

    ax.set_title(f"Frame {frame_idx+1}/{len(frames)}")
    return skeleton_lines + [scatter_plot]

# ------------- MAIN -------------
frames = load_pose_frames(CSV_PATH)
print(f"Loaded {len(frames)} frames from {CSV_PATH}")

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')

# Adjust your axes limits to match your data range
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)  # invert Y if 0 is top

# Create a line2D for each skeleton edge
skeleton_lines = []
for _ in POSE_CONNECTIONS:
    line, = ax.plot([], [], 'r-', lw=2)
    skeleton_lines.append(line)

# Create a scatter for the keypoints
scatter_plot = ax.scatter([], [], c='blue', s=30)

ani = animation.FuncAnimation(
    fig,
    update_func,         # called for each frame
    frames=len(frames),  # total number of frames
    init_func=init_func, # initialization
    interval=50,         # delay between frames in ms (20 fps if 50ms)
    blit=True            # improves performance
)

plt.show()
