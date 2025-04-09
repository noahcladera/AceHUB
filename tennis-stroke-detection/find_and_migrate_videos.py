#!/usr/bin/env python3
import os
import shutil
import glob

# Define paths
DATA_DIR = "data"
VIDEOS_DIR = "videos"

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def find_files_by_pattern(pattern):
    """Find files matching a glob pattern"""
    return glob.glob(pattern, recursive=True)

def migrate_video(video_name):
    """Find and migrate all files for a specific video"""
    # Create destination directory
    dest_video_dir = os.path.join(VIDEOS_DIR, video_name)
    ensure_directory_exists(dest_video_dir)
    
    # Files to look for with their source and destination paths
    file_patterns = [
        # Video MP4 files
        {
            "pattern": f"{DATA_DIR}/*/{video_name}/{video_name}.mp4",
            "dest_name": f"{video_name}.mp4"
        },
        # Normalized CSV files
        {
            "pattern": f"{DATA_DIR}/*/{video_name}/{video_name}_normalized.csv",
            "dest_name": f"{video_name}_normalized.csv"
        },
        # Data CSV files
        {
            "pattern": f"{DATA_DIR}/*/{video_name}/{video_name}_data.csv",
            "dest_name": f"{video_name}_data.csv"
        },
        # LLC files (handle both formats: video_X.llc and video_X.mp4.llc)
        {
            "pattern": f"{DATA_DIR}/*/{video_name}/{video_name}.llc",
            "dest_name": f"{video_name}.llc"
        },
        {
            "pattern": f"{DATA_DIR}/*/{video_name}/{video_name}.mp4.llc",
            "dest_name": f"{video_name}.llc"
        },
        # Labeled CSV files
        {
            "pattern": f"{DATA_DIR}/*/{video_name}/{video_name}_labeled.csv",
            "dest_name": f"{video_name}_labeled.csv"
        }
    ]
    
    # Track which files were copied and which are missing
    copied_files = []
    missing_files = []
    
    for file_def in file_patterns:
        found_files = find_files_by_pattern(file_def["pattern"])
        
        if found_files:
            # Take the first matching file if multiple found
            source_path = found_files[0]
            dest_path = os.path.join(dest_video_dir, file_def["dest_name"])
            
            # Copy the file if it doesn't exist or force overwrite
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                copied_files.append(file_def["dest_name"])
                print(f"Copied {source_path} -> {dest_path}")
            else:
                print(f"Skipped {source_path} (already exists)")
        else:
            missing_files.append(file_def["dest_name"])
    
    # Check if clips folder exists and has contents
    clips_pattern = f"{DATA_DIR}/*/{video_name}/{video_name}_clips"
    clips_folders = glob.glob(clips_pattern)
    
    has_clips = False
    if clips_folders:
        clips_source = clips_folders[0]
        if os.path.isdir(clips_source) and os.listdir(clips_source):
            has_clips = True
            clips_dest = os.path.join(dest_video_dir, f"{video_name}_clips")
            
            # Copy the clips folder if it doesn't exist
            if not os.path.exists(clips_dest):
                ensure_directory_exists(clips_dest)
                
                # Copy each clip
                for clip_file in os.listdir(clips_source):
                    src_clip = os.path.join(clips_source, clip_file)
                    dst_clip = os.path.join(clips_dest, clip_file)
                    
                    if os.path.isfile(src_clip) and not os.path.exists(dst_clip):
                        shutil.copy2(src_clip, dst_clip)
                        print(f"Copied clip {src_clip} -> {dst_clip}")
    
    # Create status file
    status = {
        "has_video": f"{video_name}.mp4" in copied_files or os.path.exists(os.path.join(dest_video_dir, f"{video_name}.mp4")),
        "has_normalized_csv": f"{video_name}_normalized.csv" in copied_files or os.path.exists(os.path.join(dest_video_dir, f"{video_name}_normalized.csv")),
        "has_data_csv": f"{video_name}_data.csv" in copied_files or os.path.exists(os.path.join(dest_video_dir, f"{video_name}_data.csv")),
        "has_llc": f"{video_name}.llc" in copied_files or os.path.exists(os.path.join(dest_video_dir, f"{video_name}.llc")),
        "has_clips": has_clips or os.path.exists(os.path.join(dest_video_dir, f"{video_name}_clips")),
        "is_ready_for_clipping": False,
        "is_fully_processed": False
    }
    
    # Calculate ready for clipping status
    status["is_ready_for_clipping"] = (
        status["has_video"] and 
        status["has_normalized_csv"] and 
        status["has_llc"]
    )
    
    # Calculate fully processed status
    status["is_fully_processed"] = status["has_clips"]
    
    # Write status file
    status_path = os.path.join(dest_video_dir, "status.txt")
    with open(status_path, 'w') as f:
        for key, value in status.items():
            f.write(f"{key}: {value}\n")
    
    summary = {
        "video_name": video_name,
        "copied_files": copied_files,
        "missing_files": missing_files,
        "status": status
    }
    
    return summary

def get_all_video_names():
    """Find all video folders that exist in the data directory"""
    video_names = set()
    
    # Look for video folders in different subdirectories
    patterns = [
        f"{DATA_DIR}/*/video_*",
        f"{DATA_DIR}/*/*/video_*"
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                video_name = os.path.basename(path)
                if video_name.startswith("video_"):
                    video_names.add(video_name)
    
    return sorted(list(video_names))

def main():
    """Find and migrate all video files"""
    print(f"=== VIDEO FILES MIGRATION ===")
    print(f"Finding videos in {DATA_DIR} and migrating to {VIDEOS_DIR}")
    
    # Make sure the videos directory exists
    ensure_directory_exists(VIDEOS_DIR)
    
    # Find all video names
    video_names = get_all_video_names()
    print(f"Found {len(video_names)} videos to process")
    
    # Stats tracking
    stats = {
        "total": len(video_names),
        "migrated": 0,
        "ready_for_clipping": 0,
        "fully_processed": 0
    }
    
    # Process each video
    for video_name in video_names:
        print(f"\nProcessing {video_name}...")
        summary = migrate_video(video_name)
        
        stats["migrated"] += 1
        if summary["status"]["is_ready_for_clipping"]:
            stats["ready_for_clipping"] += 1
        if summary["status"]["is_fully_processed"]:
            stats["fully_processed"] += 1
        
        # Print quick summary for this video
        status_msg = ""
        if summary["status"]["is_fully_processed"]:
            status_msg = "FULLY PROCESSED"
        elif summary["status"]["is_ready_for_clipping"]:
            status_msg = "READY FOR CLIPPING"
        else:
            status_msg = "MISSING FILES"
        
        print(f"Status: {status_msg}")
        print(f"  - Copied: {', '.join(summary['copied_files']) if summary['copied_files'] else 'None'}")
        print(f"  - Missing: {', '.join(summary['missing_files']) if summary['missing_files'] else 'None'}")
    
    # Print final summary
    print("\n=== MIGRATION SUMMARY ===")
    print(f"Total videos found: {stats['total']}")
    print(f"Videos migrated: {stats['migrated']}")
    print(f"Videos ready for clipping: {stats['ready_for_clipping']}")
    print(f"Videos fully processed: {stats['fully_processed']}")
    print("\nNext steps:")
    print("1. Run the reorganize_videos.py script if you have more videos to add")
    print("2. Run the stroke_segmentation.py script to process videos that are ready for clipping")
    print("3. Check the Strokes_Library directory for processed stroke clips")

if __name__ == "__main__":
    main() 