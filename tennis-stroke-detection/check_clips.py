#!/usr/bin/env python3
import os
import re
from collections import defaultdict

# Define paths
input_dir = "Strokes_Library"

# Get all files in the input directory
if os.path.exists(input_dir):
    dirs = os.listdir(input_dir)
    stroke_folders = [d for d in dirs if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("stroke_")]
    
    # Count the number of stroke folders
    print(f"Total number of stroke folders: {len(stroke_folders)}")
    
    # Check file counts in each stroke folder
    file_counts = {}
    for folder in stroke_folders:
        folder_path = os.path.join(input_dir, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        count = len(files)
        file_counts[count] = file_counts.get(count, 0) + 1
    
    print("Breakdown by number of files per stroke folder:")
    for count, num_folders in sorted(file_counts.items()):
        print(f"  Folders with {count} files: {num_folders}")
    
    # Check for missing files in stroke folders
    expected_files = ["stroke.csv", "stroke_norm.csv", "stroke_clip.mp4", 
                      "stroke_overlay.mp4", "stroke_skeleton.mp4"]
    
    folders_with_missing_files = []
    for folder in stroke_folders:
        folder_path = os.path.join(input_dir, folder)
        files = os.listdir(folder_path)
        
        missing = [exp_file for exp_file in expected_files if exp_file not in files]
        if missing:
            folders_with_missing_files.append((folder, missing))
    
    if folders_with_missing_files:
        print("\nStroke folders with missing files:")
        for folder, missing in folders_with_missing_files[:5]:  # Show just the first 5
            print(f"  {folder} missing: {', '.join(missing)}")
        
        if len(folders_with_missing_files) > 5:
            print(f"  ... and {len(folders_with_missing_files) - 5} more folders with missing files.")
    else:
        print("\nAll stroke folders have the expected 5 files.")
else:
    print(f"Directory not found: {input_dir}")
    print("Please run reorganize_library.py first to create the Strokes_Library.") 