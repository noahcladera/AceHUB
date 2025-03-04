#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimized_batch_download_and_convert.py

Downloads and converts multiple YouTube videos to 30 FPS concurrently, with FPS checks to skip re-encode if unnecessary.
"""

import os
import yt_dlp
import ffmpeg
import concurrent.futures

# ------------------------------------------
# CONFIGURABLE SETTINGS
# ------------------------------------------
ACEHUB_FOLDER = "acehub"
DATA_FOLDER = os.path.join(ACEHUB_FOLDER, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# List of YouTube URLs
YOUTUBE_URLS = [
    # Add your video URLs here
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
    "https://www.youtube.com/watch?v=fhn2ANDE4kA&ab_channel=Slow-MoTennis",
]

# Desired frame rate
FPS_TARGET = 30

# Number of worker processes for parallel downloads/encodes
MAX_WORKERS = 4

# Set to True to try using NVIDIA hardware encoder (requires a supported GPU, drivers, etc.)
USE_NVIDIA_HARDWARE = False  # Set to False if you want CPU-based x264

# ------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------

def download_video(url, idx, video_folder):
    """
    Downloads the best available video from YouTube using yt-dlp.
    Saves the file as "video_{idx}_temp.mp4" in the provided folder.
    Returns the path to the downloaded file.
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
    print(f"[INFO] Downloaded temp video: {outtmpl}")
    return outtmpl

def get_video_fps(video_path):
    """
    Returns the average FPS of the given video, or None if not found.
    """
    try:
        probe = ffmpeg.probe(video_path)
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                # avg_frame_rate can be a fraction, e.g. "30000/1001"
                # so split and handle that
                fps_str = stream.get("avg_frame_rate", "0/0")
                num, denom = fps_str.split('/')
                if float(denom) != 0:
                    return float(num) / float(denom)
    except ffmpeg.Error as e:
        print(f"[ERROR] Could not probe video: {video_path} -> {e}")
    return None

def convert_video_to_30fps(input_path, output_path, fps):
    """
    Converts the input video to the specified FPS using FFmpeg.
    Re-encodes video. By default uses x264 'ultrafast' preset for speed.
    If USE_NVIDIA_HARDWARE is True and you have an NVIDIA GPU, uses h264_nvenc.
    """
    print(f"[INFO] Converting {input_path} to {fps} FPS...")

    if USE_NVIDIA_HARDWARE:
        # NVIDIA GPU hardware-accelerated encode
        vcodec = "h264_nvenc"
        preset = "fast"  # or "p1" / "hp" for NVENC
        # You can also add flags for GPU usage, etc.
    else:
        # CPU-based x264
        vcodec = "libx264"
        preset = "ultrafast"

    try:
        stream = (
            ffmpeg
            .input(input_path)
            .filter('fps', fps=fps, round='up')
            .output(
                output_path,
                vcodec=vcodec,
                preset=preset,
                crf=23,           # adjust CRF if you want smaller/larger file
                acodec='copy',
                overwrite_output=True
            )
        )
        ffmpeg.run(stream, quiet=True)
        print(f"[INFO] Finished converting to {fps} FPS: {output_path}")
    except ffmpeg.Error as e:
        print(f"[ERROR] FFmpeg conversion failed for {input_path}: {e}")

def process_single_video(idx, url):
    """
    End-to-end process for a single video:
    1. Create a folder for the video
    2. Download the video to a temp file
    3. Check if FPS is already 30; if so, just rename/copy
    4. Otherwise, re-encode to 30 FPS
    5. Clean up temp file
    """
    video_folder = os.path.join(DATA_FOLDER, f"video_{idx}")
    os.makedirs(video_folder, exist_ok=True)

    # 1) Download
    temp_video_path = download_video(url, idx, video_folder)

    # 2) Probe FPS
    original_fps = get_video_fps(temp_video_path)
    if original_fps:
        print(f"[INFO] Original FPS = {original_fps:.2f}")

    # 3) Define final output path
    output_video_path = os.path.join(video_folder, f"video_{idx}.mp4")

    # 4) If the video is already close to 30 FPS, skip re-encode
    if original_fps and abs(original_fps - FPS_TARGET) < 0.1:
        print(f"[INFO] Video {idx} is already ~{FPS_TARGET} FPS, skipping re-encode.")
        # Just rename the temp file to final
        os.rename(temp_video_path, output_video_path)
    else:
        # Re-encode to 30 FPS
        convert_video_to_30fps(temp_video_path, output_video_path, FPS_TARGET)
        # Remove the temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"[INFO] Removed temp file: {temp_video_path}")

    return f"[DONE] Video {idx} processed -> {output_video_path}"

def main():
    print(">>> Starting concurrent batch download + 30 FPS conversion...")

    # Using multiple processes to parallelize
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each video’s work
        futures = [
            executor.submit(process_single_video, idx, url)
            for idx, url in enumerate(YOUTUBE_URLS, start=1)
        ]

        # As each future completes, print the result
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print(">>> All tasks complete.")

if __name__ == "__main__":
    main()
