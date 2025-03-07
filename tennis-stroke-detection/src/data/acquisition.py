#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
acquisition.py

This script downloads videos from YouTube using yt-dlp and then converts them
to 30 FPS using FFmpeg. Each video is saved in a subfolder under
"tennis-stroke-detection/data/raw". It checks if videos already exist
before downloading, to avoid overwriting existing work.
"""

import os
import yt_dlp
import ffmpeg

# Path setup: we assume this script lives in tennis-stroke-detection/src/data/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw")
os.makedirs(DATA_FOLDER, exist_ok=True)

# List of YouTube URLs to process
youtube_urls = [
    # ... your YouTube URLs ...
]

FPS = 30  # Desired frame rate for converted videos


def download_video(url, idx, video_folder):
    """
    Downloads the best available video from YouTube using yt-dlp.
    Saves the temporary file as "video_{idx}_temp.mp4" in the provided folder.
    """
    temp_filename = f"video_{idx}_temp.mp4"
    outtmpl = os.path.join(video_folder, temp_filename)
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"[INFO] Downloaded temporary video: {outtmpl}")
    return outtmpl

def convert_video_to_30fps(input_path, output_path, fps):
    """
    Converts the input video to the specified FPS using FFmpeg.
    Re-encodes video with libx264 and copies the audio.
    """
    try:
        stream = ffmpeg.input(input_path)
        stream = stream.filter('fps', fps=fps, round='up')
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec='libx264',
            crf=23,
            preset='fast',
            acodec='copy'
        )
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        print(f"[INFO] Converted video to {fps} FPS: {output_path}")
    except ffmpeg.Error as e:
        print("Error converting video to 30 FPS:", e)

def find_next_available_index():
    """
    Finds the next available video index by checking existing 'video_X' folders
    in the raw data directory. Returns the highest existing index + 1.
    """
    max_index = 0
    for item in os.listdir(DATA_FOLDER):
        if os.path.isdir(os.path.join(DATA_FOLDER, item)) and item.startswith("video_"):
            try:
                index = int(item.split("_")[1])
                max_index = max(max_index, index)
            except (ValueError, IndexError):
                pass
    return max_index + 1

def main():
    print(">>> Starting batch download and conversion of videos...")

    start_index = find_next_available_index()
    print(f"[INFO] Starting from video index {start_index} (based on existing folders)")

    for i, url in enumerate(youtube_urls):
        idx = start_index + i
        print(f"[INFO] Processing video {idx}: {url}")

        # Create a subfolder for this video under data/raw
        video_folder = os.path.join(DATA_FOLDER, f"video_{idx}")
        final_video_path = os.path.join(video_folder, f"video_{idx}.mp4")

        # If already present, skip
        if os.path.exists(final_video_path):
            print(f"[INFO] Video {idx} already exists at {final_video_path}. Skipping.")
            continue

        os.makedirs(video_folder, exist_ok=True)

        # 1) Download the video
        temp_video_path = download_video(url, idx, video_folder)

        # 2) Convert the downloaded video to 30 FPS
        convert_video_to_30fps(temp_video_path, final_video_path, FPS)

        # 3) Remove the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"[INFO] Removed temporary file: {temp_video_path}")

    print(">>> Batch download and conversion complete.")

if __name__ == "__main__":
    main()