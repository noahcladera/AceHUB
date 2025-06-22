#!/usr/bin/env python3
"""
extract_csv.py  —  Convert every raw video in
    <repo_root>/unprocessed_videos/
(except anything inside <repo_root>/unprocessed_videos/archive/)
into a single MediaPipe-Pose landmark CSV placed in
    <repo_root>/Thesis/video_csv/

Run from anywhere:
    python tennis-stroke-detection/Thesis/extract_csv.py
"""

import csv, sys, cv2
from pathlib import Path
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ----------------------------------------------------------------------
# Resolve project structure automatically
THIS_FILE  = Path(__file__).resolve()
REPO_ROOT  = THIS_FILE.parents[1]                 # tennis-stroke-detection/
RAW_DIR    = REPO_ROOT / "unprocessed_videos"
OUT_DIR    = THIS_FILE.parent / "video_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------------------------------------

# ---------------------------- helpers --------------------------------
def header():
    hdr = ["frame_index"]
    for i in range(33):
        hdr += [f"lm_{i}_{ax}" for ax in ("x", "y", "z", "vis")]
    hdr += ["right_elbow_angle", "left_elbow_angle", "stroke_label"]
    return hdr

def elbow_angle(ax, ay, bx, by, cx, cy):
    BA = np.array([ax - bx, ay - by])
    BC = np.array([cx - bx, cy - by])
    cos = (BA @ BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def process_video(video_path: Path) -> None:
    out_csv = OUT_DIR / f"{video_path.stem}_data.csv"
    if out_csv.exists():
        print(f"[SKIP] {out_csv.name} already there")
        return

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=2,
        smooth_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERR] cannot open {video_path.name}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header())

        pbar = tqdm(total=total, desc=video_path.name, unit="f")
        idx  = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            row = [idx]
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for lm in lms:
                    row += [lm.x, lm.y,
                            getattr(lm, "z", 0.0),
                            getattr(lm, "visibility", 1.0)]
                # 2-D elbow angles
                l_angle = elbow_angle(lms[11].x, lms[11].y,
                                      lms[13].x, lms[13].y,
                                      lms[15].x, lms[15].y)
                r_angle = elbow_angle(lms[12].x, lms[12].y,
                                      lms[14].x, lms[14].y,
                                      lms[16].x, lms[16].y)
            else:
                row += [0.0] * 33 * 4
                l_angle = r_angle = 0.0
            row += [r_angle, l_angle, 1]   # dummy stroke_label
            writer.writerow(row)

            idx += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    mp_pose.close()
    print(f"[✓] {out_csv.relative_to(REPO_ROOT)}")

# --------------------------- main loop -------------------------------
def main():
    if not RAW_DIR.exists():
        sys.exit(f"[ERR] {RAW_DIR} does not exist")

    videos = [p for p in RAW_DIR.glob("*")
              if p.suffix.lower() in (".mp4", ".mov", ".avi")
              and "archive" not in p.parts]     # ignore archive sub-folder

    if not videos:
        print("[INFO] No videos found in unprocessed_videos/")
        return

    for vid in sorted(videos):
        process_video(vid)

if __name__ == "__main__":
    main()
