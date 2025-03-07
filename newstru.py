import os
import shutil

def copy_script(src_file, dst_file):
    """
    Copies src_file to dst_file, creating directories for dst_file if needed.
    """
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)
    print(f"Copied: {src_file} -> {dst_file}")

def main():
    # Where your old scripts currently reside
    old_scripts_path = "acehub-old/scripts"
    # Where your new structured project resides
    new_project_path = "tennis-stroke-detection"

    # Map each old script to its new target path in the new structure
    file_map = {
        # old script name -> new location (under "src/data" or "src/inference", etc.)
        "1Batch_download_videos.py": os.path.join(new_project_path, "src", "data", "acquisition.py"),
        "2 2Parallel_batch_process.py": os.path.join(new_project_path, "src", "data", "pose_extraction.py"),
        "3Full_Video_Normalization.py": os.path.join(new_project_path, "src", "data", "normalization.py"),
        "4Create_Frame_Labels.py": os.path.join(new_project_path, "src", "data", "feature_engineering.py"),
        "5Create_clips_from_cuts.py": os.path.join(new_project_path, "src", "inference", "clip_generator.py")
    }

    # Copy each script from old to new location
    for old_script, new_location in file_map.items():
        src_file = os.path.join(old_scripts_path, old_script)
        if not os.path.isfile(src_file):
            print(f"[ERROR] Could not find source script: {src_file}")
            continue

        copy_script(src_file, new_location)

    print("\nScripts migrated successfully!")

if __name__ == "__main__":
    main()