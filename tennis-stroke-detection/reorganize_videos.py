#!/usr/bin/env python3
import os
import shutil
import glob

# Define paths
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim") 
VIDEOS_DIR = "videos"  # New top-level directory for all videos

def create_directory_structure():
    """Create the new directory structure if it doesn't exist"""
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    print(f"Created videos directory: {VIDEOS_DIR}")

def find_video_files():
    """Find all video_X folders in the processed directory"""
    if not os.path.exists(PROCESSED_DIR):
        print(f"Error: Processed directory not found: {PROCESSED_DIR}")
        return []
    
    video_folders = []
    for item in os.listdir(PROCESSED_DIR):
        if os.path.isdir(os.path.join(PROCESSED_DIR, item)) and item.startswith("video_"):
            video_folders.append(item)
    
    print(f"Found {len(video_folders)} video folders in {PROCESSED_DIR}")
    return video_folders

def reorganize_video(video_name):
    """Reorganize files for a single video"""
    # Source paths
    processed_video_dir = os.path.join(PROCESSED_DIR, video_name)
    interim_video_dir = os.path.join(INTERIM_DIR, video_name)
    
    # Destination path
    dest_video_dir = os.path.join(VIDEOS_DIR, video_name)
    os.makedirs(dest_video_dir, exist_ok=True)
    
    # Files to look for
    video_mp4 = os.path.join(processed_video_dir, f"{video_name}.mp4")
    normalized_csv = os.path.join(processed_video_dir, f"{video_name}_normalized.csv")
    llc_file = os.path.join(processed_video_dir, f"{video_name}.llc")
    data_csv = os.path.join(interim_video_dir, f"{video_name}_data.csv")
    
    # Track what we copied and what's missing
    copied_files = []
    missing_files = []
    
    # Copy files if they exist
    if os.path.exists(video_mp4):
        dest_path = os.path.join(dest_video_dir, f"{video_name}.mp4")
        shutil.copy2(video_mp4, dest_path)
        copied_files.append(f"{video_name}.mp4")
    else:
        missing_files.append(f"{video_name}.mp4")
    
    if os.path.exists(normalized_csv):
        dest_path = os.path.join(dest_video_dir, f"{video_name}_normalized.csv")
        shutil.copy2(normalized_csv, dest_path)
        copied_files.append(f"{video_name}_normalized.csv")
    else:
        missing_files.append(f"{video_name}_normalized.csv")
    
    if os.path.exists(llc_file):
        dest_path = os.path.join(dest_video_dir, f"{video_name}.llc")
        shutil.copy2(llc_file, dest_path)
        copied_files.append(f"{video_name}.llc")
    else:
        missing_files.append(f"{video_name}.llc")
    
    if os.path.exists(data_csv):
        dest_path = os.path.join(dest_video_dir, f"{video_name}_data.csv")
        shutil.copy2(data_csv, dest_path)
        copied_files.append(f"{video_name}_data.csv")
    else:
        missing_files.append(f"{video_name}_data.csv")
    
    # Check if we found any clips for this video
    clips_folder = os.path.join(processed_video_dir, f"{video_name}_clips")
    has_clips = os.path.exists(clips_folder) and len(os.listdir(clips_folder)) > 0
    
    # Create status file to track whether this video has been processed for strokes
    status = {
        "has_video": f"{video_name}.mp4" in copied_files,
        "has_normalized_csv": f"{video_name}_normalized.csv" in copied_files,
        "has_llc": f"{video_name}.llc" in copied_files,
        "has_data_csv": f"{video_name}_data.csv" in copied_files,
        "has_clips": has_clips,
        "is_ready_for_clipping": (f"{video_name}.mp4" in copied_files and 
                                f"{video_name}_normalized.csv" in copied_files and 
                                f"{video_name}.llc" in copied_files),
        "is_fully_processed": has_clips
    }
    
    # Write status to a simple text file
    status_txt = os.path.join(dest_video_dir, "status.txt")
    with open(status_txt, 'w') as f:
        for key, value in status.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Reorganized {video_name}:")
    print(f"  - Copied: {', '.join(copied_files)}")
    if missing_files:
        print(f"  - Missing: {', '.join(missing_files)}")
    print(f"  - Status: {'Ready for clipping' if status['is_ready_for_clipping'] and not status['is_fully_processed'] else 'Already processed' if status['is_fully_processed'] else 'Missing required files'}")
    
    return status

def main():
    """Main function to reorganize all videos"""
    print("=== VIDEO REORGANIZATION SCRIPT ===")
    
    # Create new directory structure
    create_directory_structure()
    
    # Find all video folders
    video_folders = find_video_files()
    
    # Track overall statistics
    stats = {
        "total": 0,
        "ready_for_clipping": 0,
        "already_processed": 0,
        "missing_files": 0
    }
    
    # Reorganize each video
    for video_name in video_folders:
        status = reorganize_video(video_name)
        stats["total"] += 1
        
        if status["is_fully_processed"]:
            stats["already_processed"] += 1
        elif status["is_ready_for_clipping"]:
            stats["ready_for_clipping"] += 1
        else:
            stats["missing_files"] += 1
    
    # Print summary
    print("\n=== REORGANIZATION SUMMARY ===")
    print(f"Total videos: {stats['total']}")
    print(f"Videos ready for clipping: {stats['ready_for_clipping']}")
    print(f"Videos already processed: {stats['already_processed']}")
    print(f"Videos missing required files: {stats['missing_files']}")
    print("\nTo process videos that are ready for clipping, run:")
    print("python src/stroke_segmentation.py")

if __name__ == "__main__":
    main() 