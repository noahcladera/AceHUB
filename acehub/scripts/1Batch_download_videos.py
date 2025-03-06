#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1Batch_download_and_Convert_Videos.py

This script downloads videos from YouTube using yt-dlp and then converts them to 30 FPS using FFmpeg.
Each video is saved in its respective folder under the "acehub/data" directory.
"""

import os
import yt_dlp
import ffmpeg

# Path to the top-level acehub folder
ACEHUB_FOLDER = "acehub"

# Folder for data inside acehub
DATA_FOLDER = os.path.join(ACEHUB_FOLDER, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# List of YouTube URLs to process
youtube_urls = [
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
       # Add more URLs as needed...
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
        # Force FPS to 'fps'
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

def main():
    print(">>> Starting batch download and conversion of videos...")
    for idx, url in enumerate(youtube_urls, start=1):
        print(f"[INFO] Processing video {idx}: {url}")
        # Create a folder for the current video inside acehub/data
        video_folder = os.path.join(DATA_FOLDER, f"video_{idx}")
        os.makedirs(video_folder, exist_ok=True)

        # 1) Download the video to a temporary file
        temp_video_path = download_video(url, idx, video_folder)

        # 2) Define the final output path: video_{idx}.mp4 in the same folder
        output_video_path = os.path.join(video_folder, f"video_{idx}.mp4")

        # 3) Convert the downloaded video to 30 FPS
        convert_video_to_30fps(temp_video_path, output_video_path, FPS)

        # 4) Remove the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"[INFO] Removed temporary file: {temp_video_path}")

    print(">>> Batch download and conversion complete.")

if __name__ == "__main__":
    main()
