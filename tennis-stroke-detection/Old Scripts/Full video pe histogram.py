
import os
import re
import csv
import math
import cv2
import mediapipe as mp

##############################################################################
#APPROACH
##############################################################################
# : Extract Full Video Pose Data, THEN Slice into Strokes
#   1. For each video, run pose estimation (frame by frame) and save full raw data.
#   2. Post-process that data offline to detect strokes (e.g., racket-back to contact).
#   3. Once strokes are identified by frame range, slice the ORIGINAL video using FFmpeg.


##############################################################################
# CONFIGURATION SETTINGS
##############################################################################
VIDEOS_DIR = "tennis_clips/videos"  # Folder containing video_*.mp4
VIDEOS_DIR = "Test_media/test_videos"  # Folder containing video_*.mp4
CALCULATE_ANGLES = True            # Set to True to calculate extra joint angles

# Set up MediaPipe Pose (BlazePose Full)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Use 2 for full-body (BlazePose Full)
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

##############################################################################
# HELPER FUNCTIONS
##############################################################################
def get_video_number(filename):
    """
    Extracts a numeric index from a filename like 'video_10.mp4' => 10 for sorting.
    """
    match = re.search(r'video_(\d+)\.mp4', filename)
    return int(match.group(1)) if match else 999999

def calculate_angle(ax, ay, bx, by, cx, cy):
    """
    Calculate the angle at point B formed by A->B->C in 2D space (x,y).
    """
    BAx = ax - bx
    BAy = ay - by
    BCx = cx - bx
    BCy = cy - by
    dot = BAx * BCx + BAy * BCy
    magBA = math.sqrt(BAx**2 + BAy**2)
    magBC = math.sqrt(BCx**2 + BCy**2)

    if magBA == 0 or magBC == 0:
        return 0.0

    cos_angle = dot / (magBA * magBC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # clamp to avoid domain errors
    return math.degrees(math.acos(cos_angle))

def process_video(video_path):
    """
    Opens a single video (e.g., 'video_1.mp4'), runs MediaPipe Pose on every frame,
    and saves landmark data (+ optional angles) in 'video_1_data.csv' (same folder).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(VIDEOS_DIR, f"{video_name}_data.csv")

    # Overwrite any existing CSV file
    if os.path.exists(output_csv):
        os.remove(output_csv)

    print(f"Processing {video_name} -> {output_csv}")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header: frame_index + 33 landmarks * (x,y,z,vis) + optional angles
        header = ["frame_index"]
        for i in range(33):
            header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"]
        if CALCULATE_ANGLES:
            header += ["right_elbow_angle", "left_elbow_angle"]
        writer.writerow(header)
        print("Header written:", header)

        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                row = [frame_index]
                landmarks = results.pose_landmarks.landmark

                # Append 33 (x,y,z,visibility) sets
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.z, lm.visibility]

                # Optionally calculate elbow angles
                if CALCULATE_ANGLES:
                    r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                    l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]

                    r_angle = calculate_angle(
                        r_shoulder.x, r_shoulder.y,
                        r_elbow.x,    r_elbow.y,
                        r_wrist.x,    r_wrist.y
                    )
                    l_angle = calculate_angle(
                        l_shoulder.x, l_shoulder.y,
                        l_elbow.x,    l_elbow.y,
                        l_wrist.x,    l_wrist.y
                    )
                    row += [r_angle, l_angle]

                writer.writerow(row)

            frame_index += 1

    cap.release()
    print(f"Done with {video_name} -> {output_csv}")

##############################################################################
# MAIN SCRIPT
##############################################################################
def main():
    videos = [f for f in os.listdir(VIDEOS_DIR)
              if f.startswith("video_") and f.endswith(".mp4")]
    videos = sorted(videos, key=get_video_number)

    if not videos:
        print(f"No video_*.mp4 files found in '{VIDEOS_DIR}'!")
        return

    print("Found videos:", videos)
    for video_file in videos:
        video_path = os.path.join(VIDEOS_DIR, video_file)
        process_video(video_path)

if __name__ == "__main__":
    main()
