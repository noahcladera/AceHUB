# Unprocessed Videos Directory

This directory is where you can add new tennis videos for processing. The unified processing script will take care of all the steps from raw video to final skeleton visualizations.

## How to add a new video

1. Place your MP4 or MOV video file in this directory.

2. Run the unified processing script with your video filename:
```bash
python src/personal/process_video_unified.py your_video_file.mp4
# OR
python src/personal/process_video_unified.py your_video_file.mov
```

3. The script will extract pose data and prompt you to edit an LLC file to mark tennis strokes. The LLC file will be created in this directory with the same name as your video.

4. Edit the LLC file to mark the start and end times of each tennis stroke using the following format:
```
start_time end_time stroke_type
```

Example:
```
10.5 15.2 forehand
20.1 23.8 backhand
30.5 34.0 serve
```

5. After you've edited the LLC file, press Enter to continue the processing.

6. The script will complete the remaining steps automatically and generate video clips in the `Strokes_Library` directory.

## Advanced Usage

You can skip specific processing steps by using the `--skip-steps` parameter:

```bash
python src/personal/process_video_unified.py your_video_file.mp4 --skip-steps pose,norm,llc,3d
```

You can also specify a custom video ID instead of using the next available one:

```bash
python src/personal/process_video_unified.py your_video_file.mp4 --video-id 80
```

## Output

After processing is complete, you'll find:
- Video clips in the `Strokes_Library` directory
- Each clip will include raw video, skeleton overlay video, and skeleton-only video
- Interactive 3D visualizations of the skeleton data for each stroke

The script handles video format conversion automatically, supporting both MP4 and MOV files. 