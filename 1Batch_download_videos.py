import cv2
import numpy as np
import yt_dlp
import ffmpeg
import os
import sys

#############################################
# CONFIGURATION & FOLDER SETUP
#############################################
# Base folder for all outputs
BASE_FOLDER = "tennis_clips"
os.makedirs(BASE_FOLDER, exist_ok=True)

# Folder for downloaded videos
VIDEOS_FOLDER = os.path.join(BASE_FOLDER, "videos")
os.makedirs(VIDEOS_FOLDER, exist_ok=True)


# List of YouTube URLs to process (batch download)
youtube_urls = [
    
    "https://www.youtube.com/watch?v=Nw_2I2ksX3U&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=4&ab_channel=LoveTennis",
    
    
]


#############################################
# STEP 1: BATCH DOWNLOAD OF VIDEOS
#############################################
def download_video(url, idx):
    """
    Downloads the best available video (MP4) from YouTube by merging best video and audio streams.
    Saves the video in VIDEOS_FOLDER with a unique filename.
    """
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(VIDEOS_FOLDER, f"video_{idx}.%(ext)s"),
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])



#############################################
# MAIN EXECUTION
#############################################
if __name__ == "__main__":
    print(">>> Starting batch download of videos...")
    # Download each video in the list with a unique index
    for idx, url in enumerate(youtube_urls, start=1):
        print(f"Downloading video {idx}: {url}")
        download_video(url, idx)
    print("Batch download complete. Downloaded videos saved to:", VIDEOS_FOLDER)

