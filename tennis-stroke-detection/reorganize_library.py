#!/usr/bin/env python3
import os
import shutil
import re
from collections import defaultdict

# Define paths
input_dir = "Final library"
output_dir = "Strokes_Library"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all files in the input directory
files = os.listdir(input_dir)

# Filter out non-files (like .DS_Store)
files = [f for f in files if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]

# Group files by video and clip number
clips = defaultdict(list)
for file in files:
    # Extract video and clip numbers using regex
    match = re.match(r'video_(\d+)_clip_(\d+)', file)
    if match:
        video_num = match.group(1)
        clip_num = match.group(2)
        key = f"{video_num}_{clip_num}"
        clips[key].append(file)

# Fix: The issue was that we were only processing clips if they had exactly 5 files,
# but we need to process clips with any sufficient set of files.
stroke_counter = 1
total_clips = len(clips)
print(f"Found {total_clips} unique clips to process")

for key, clip_files in clips.items():
    # Create stroke directory
    stroke_dir = os.path.join(output_dir, f"stroke_{stroke_counter}")
    if not os.path.exists(stroke_dir):
        os.makedirs(stroke_dir)
    
    # Copy and rename files
    has_missing_files = False
    for file_type in ["csv", "norm_csv", "mp4", "overlay_mp4", "skeleton_mp4"]:
        file_found = False
        
        for file in clip_files:
            # Match file type with the appropriate file
            if file_type == "csv" and file.endswith(".csv") and not file.endswith("_norm.csv"):
                source_path = os.path.join(input_dir, file)
                dest_path = os.path.join(stroke_dir, "stroke.csv")
                print(f"Copying {source_path} to {dest_path}")
                shutil.copy2(source_path, dest_path)
                file_found = True
                break
            elif file_type == "norm_csv" and file.endswith("_norm.csv"):
                source_path = os.path.join(input_dir, file)
                dest_path = os.path.join(stroke_dir, "stroke_norm.csv")
                print(f"Copying {source_path} to {dest_path}")
                shutil.copy2(source_path, dest_path)
                file_found = True
                break
            elif file_type == "mp4" and file.endswith(".mp4") and not file.endswith("_skeleton.mp4") and not file.endswith("_overlay.mp4"):
                source_path = os.path.join(input_dir, file)
                dest_path = os.path.join(stroke_dir, "stroke_clip.mp4")
                print(f"Copying {source_path} to {dest_path}")
                shutil.copy2(source_path, dest_path)
                file_found = True
                break
            elif file_type == "overlay_mp4" and file.endswith("_overlay.mp4"):
                source_path = os.path.join(input_dir, file)
                dest_path = os.path.join(stroke_dir, "stroke_overlay.mp4")
                print(f"Copying {source_path} to {dest_path}")
                shutil.copy2(source_path, dest_path)
                file_found = True
                break
            elif file_type == "skeleton_mp4" and file.endswith("_skeleton.mp4"):
                source_path = os.path.join(input_dir, file)
                dest_path = os.path.join(stroke_dir, "stroke_skeleton.mp4")
                print(f"Copying {source_path} to {dest_path}")
                shutil.copy2(source_path, dest_path)
                file_found = True
                break
        
        if not file_found:
            print(f"Warning: Missing {file_type} file for clip {key}")
            has_missing_files = True
    
    stroke_counter += 1
    
    # Print progress every 50 clips
    if stroke_counter % 50 == 0:
        print(f"Processed {stroke_counter-1} out of {total_clips} clips")

print(f"Reorganization complete. Created {stroke_counter-1} stroke folders in {output_dir}") 