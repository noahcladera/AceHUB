#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
time_normalize_clips.py

For each CSV in "Final library" (e.g., "video_3_clip_1.csv"),
we resample the rows to have a fixed number of frames (RESAMPLED_FRAMES).
This solves the slow vs. normal speed issue by normalizing the timeline.

Result:
  "video_3_clip_1_norm.csv", "video_7_clip_2_norm.csv", etc.
  Each has exactly 60 frames worth of data.

Assumptions:
  - The CSV has a header row, including a "frame_index" or the first column is frame index.
  - The rest of the columns are numeric pose data, e.g. 33 landmarks × 3 coords, etc.
"""

import os
import csv
import numpy as np

FINAL_LIBRARY = "Final library"  # Folder containing your unnormalized CSVs
RESAMPLED_FRAMES = 120           # number of frames after time normalization
OUTPUT_SUFFIX = "_norm.csv"     # appended to each CSV's name

def time_normalize_csv(csv_path, out_path, num_frames):
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Convert all but the last column to float
    # The last column is swing_phase, so we ignore it during interpolation
    numeric_header = header[:-1]  # everything except 'swing_phase'
    numeric_data = []
    for row in rows:
        # e.g. row[:-1] => all but the last element
        numeric_data.append([float(x) for x in row[:-1]])
    numeric_data = np.array(numeric_data)  # shape = (F, D)

    F = numeric_data.shape[0]
    if F == 0:
        print(f"[WARN] CSV is empty: {csv_path}")
        return

    # Create new indices from 0..F-1 in 'num_frames' steps
    old_max = F - 1
    new_indices = np.linspace(0, old_max, num_frames)

    D = numeric_data.shape[1]
    new_data = []
    # Interpolate each column individually
    for col_idx in range(D):
        col = numeric_data[:, col_idx]
        col_resampled = np.interp(new_indices, np.arange(F), col)
        new_data.append(col_resampled)
    # Transpose to shape (num_frames, D)
    new_data = np.array(new_data).T

    # Build output rows
    # We'll treat the first column as the "resampled_frame_index"
    # or you can store i => 0..num_frames-1
    output_rows = []
    for i in range(num_frames):
        # The first numeric column might have been your old frame_index, so you can set your new index to i
        # or store new_data[i,0] to keep a fractional index
        row_vals = [i] + list(new_data[i,1:])  # e.g. skip the old index if you want
        output_rows.append(row_vals)

    # Adjust the header
    # We might rename the first column "resampled_frame_idx"
    new_header = ["resampled_frame_idx"] + numeric_header[1:]  # skipping the old first col name if desired

    with open(out_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(new_header)
        writer.writerows(output_rows)

    print(f"[DONE] {csv_path} -> {out_path} (skipped swing_phase)")


def main():
    # Scan the "Final library" folder for CSVs
    for filename in os.listdir(FINAL_LIBRARY):
        if not filename.lower().endswith(".csv"):
            continue
        csv_path = os.path.join(FINAL_LIBRARY, filename)
        # Output name
        base, ext = os.path.splitext(filename)
        out_filename = base + OUTPUT_SUFFIX
        out_path = os.path.join(FINAL_LIBRARY, out_filename)

        # Time-normalize
        time_normalize_csv(csv_path, out_path, RESAMPLED_FRAMES)

if __name__ == "__main__":
    main()
