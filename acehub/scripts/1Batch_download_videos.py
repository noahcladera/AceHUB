#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1Batch_download_and_Convert_Videos.py

This script downloads videos from YouTube using yt-dlp and then converts them to 30 FPS using FFmpeg.
"""

import os
import yt_dlp
import ffmpeg

#############################################
# CONFIGURATION & FOLDER SETUP
#############################################
# Base folder for all outputs
BASE_FOLDER = "tennis_clips"
os.makedirs(BASE_FOLDER, exist_ok=True)

# Folder for downloaded videos
VIDEOS_FOLDER = os.path.join(BASE_FOLDER, "videos")
os.makedirs(VIDEOS_FOLDER, exist_ok=True)

# Folder for converted videos at 30 FPS
CONVERTED_FOLDER = os.path.join(BASE_FOLDER, "videos_30fps")
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# List of YouTube URLs to process (batch download)
youtube_urls = [
"https://www.youtube.com/watch?v=lgCokC--lyI&t=235s&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=jRXUK0Jd9ZM&ab_channel=COURTLEVELTENNIS-LiamApilado",
    "https://www.youtube.com/watch?v=928wJjWeVyk&ab_channel=Slow-MoTennis",
    "https://www.youtube.com/watch?v=qkQmT4fXKD8&ab_channel=Slow-MoTennis",
    "https://www.youtube.com/watch?v=Paw1e6sIJkM&ab_channel=TennisProTV",
    "https://www.youtube.com/watch?v=MNGIcSHmmSk&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=Q7Ta9DbHKjk&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=2&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=e9qA6cXg_84&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=3&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=Nw_2I2ksX3U&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=4&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=L38Rf-DW3a4&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=6&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=EFY460oquXw&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=7&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=RDl2Kz0gd18&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=8&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=SLi2qGH418g&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=10&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=5g3PjMKxVgs&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=13&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=-DskBNOZCfo&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=15&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=AyMv2ugNc6Y&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=18&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=F6tY1LFfKuU&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=19&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=yLqmpHz2O7E&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=20&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=TBHfZkPLb30&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=21&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=20dU_JL_ME8&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=25&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=ETW2kCBcdH4&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=27&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=tqPe-IdUUa8&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=30&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=ZiigS7RtEXI&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=32&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=OvGakC-KQZs&list=PL5z1bgiPPar4j-qIDOv7VJEB1vtY1y8yP&index=33&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=bNeN2XevGLM&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=wFwidKBUt9M&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=6WyMFz4ynl4&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=SlgMvQQrYhg&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=F8Q4AvYxup0&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=RVhJBhWxFQc&t=306s&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=0edfZQPFeDc&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=AFyPREOG0BM&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=-CXrTTip_Bw&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=MTP99IHemNA&t=155s&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=A-Hcgjz1uow&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=Zam8nGoJA8o&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=1YuShuvbZnM&t=9s&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=zqc9JDC-0_Q&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=FuDJ7crbkBo&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=0tP2pmXd9Gk&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=iQQceybyG-4&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=DeynHaCn-s0&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=qVbDaYs0wT8&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=R4JKzKFNgRU&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=Z_LU2q1CROA&ab_channel=LoveTennis",
    "https://www.youtube.com/watch?v=fhn2ANDE4kA&ab_channel=Slow-MoTennis"    # Add more URLs as needed...
]

#############################################
# FUNCTIONS
#############################################
def download_video(url, idx):
    """
    Downloads the best available video (MP4) from YouTube by merging the best video and audio streams.
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
    print(f"[INFO] Downloaded video_{idx}.mp4")

def convert_video_to_30fps(input_path, output_path):
    """
    Converts the input video to 30 FPS using FFmpeg.
    Re-encodes the video with libx264 and copies the audio.
    """
    try:
        # Build the FFmpeg command pipeline:
        # - Read the input
        # - Filter to force FPS=30 (using round='up' to round frame counts)
        # - Re-encode video with libx264 (you can adjust crf/preset for quality/speed trade-off)
        # - Copy the audio stream without re-encoding
        stream = ffmpeg.input(input_path)
        stream = stream.filter('fps', fps=30, round='up')
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', crf=23, preset='fast', acodec='copy')
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        print(f"[INFO] Converted to 30FPS: {output_path}")
    except ffmpeg.Error as e:
        print("Error converting video to 30 FPS:", e)

#############################################
# MAIN EXECUTION
#############################################
if __name__ == "__main__":
    print(">>> Starting batch download and conversion of videos...")
    for idx, url in enumerate(youtube_urls, start=1):
        print(f"[INFO] Processing video {idx}: {url}")
        # Download the video
        download_video(url, idx)
        # Define input and output paths
        input_video = os.path.join(VIDEOS_FOLDER, f"video_{idx}.mp4")
        output_video = os.path.join(CONVERTED_FOLDER, f"video_{idx}_30fps.mp4")
        # Convert the video to 30 FPS
        convert_video_to_30fps(input_video, output_video)
    print(">>> Batch download and conversion complete. Converted videos saved to:", CONVERTED_FOLDER)
