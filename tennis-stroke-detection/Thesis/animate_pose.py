#!/usr/bin/env python3
"""
animate_pose.py  –  play an animated skeleton from a normalised MediaPipe-Pose CSV.

Usage:
    python tennis-stroke-detection/Thesis/animate_pose.py tennis-stroke-detection/Thesis/normalization/Procrustes/video_1_data_normalised.csv

Keys while playing:
    Space  : pause / resume
    Left   : step one frame back
    Right  : step one frame forward
    Esc    : close
"""

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# MediaPipe Pose draw-order
CONNECTIONS = [
    (11,13), (13,15), (12,14), (14,16),      # arms
    (23,25), (25,27), (24,26), (26,28),      # legs
    (11,12), (23,24), (11,23), (12,24)       # torso
]

def load_landmarks(csv_path):
    df = pd.read_csv(csv_path)
    xs = df[[f"lm_{i}_x" for i in range(33)]].values   # (F,33)
    ys = df[[f"lm_{i}_y" for i in range(33)]].values
    return xs, ys

def animate(xs, ys, fps):
    frames = len(xs)
    fig, ax = plt.subplots(figsize=(4,6))
    ax.set_aspect('equal')
    ax.invert_yaxis()
    scat = ax.scatter([], [], s=25, c='red', zorder=5)
    lines = [ax.plot([], [], c='black')[0] for _ in CONNECTIONS]
    ax.set_title("0 / %d" % (frames-1))

    def init():
        ax.set_xlim(xs.min()-0.5, xs.max()+0.5)
        ax.set_ylim(ys.max()+0.5, ys.min()-0.5)
        return scat, *lines

    def update(f):
        scat.set_offsets(np.c_[xs[f], ys[f]])
        for (a,b), ln in zip(CONNECTIONS, lines):
            ln.set_data([xs[f,a], xs[f,b]], [ys[f,a], ys[f,b]])
        ax.set_title(f"{f} / {frames-1}")
        return scat, *lines

    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, interval=1000/fps, blit=True)

    # simple keyboard controls
    paused = [False]
    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
            ani.event_source.stop() if paused[0] else ani.event_source.start()
        elif event.key == 'right':
            update(min(ani.frame_seq._iterindex+1, frames-1))
            fig.canvas.draw_idle()
        elif event.key == 'left':
            update(max(ani.frame_seq._iterindex-1, 0))
            fig.canvas.draw_idle()
        elif event.key == 'escape':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to *_normalised.csv")
    ap.add_argument("--fps", type=int, default=30, help="frames per second")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        sys.exit(f"[ERR] File not found: {csv_path}")

    xs, ys = load_landmarks(csv_path)
    animate(xs, ys, args.fps)

if __name__ == "__main__":
    main()
