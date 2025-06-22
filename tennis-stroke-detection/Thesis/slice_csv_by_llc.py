#!/usr/bin/env python3
"""
slice_csv_by_llc.py ─ cut every normalised full-video CSV into per-stroke clips
===============================================================================

Examples
--------
# Hip-centred variant
python tennis-stroke-detection/Thesis/slice_csv_by_llc.py --variant hip_norm

# Shoulder-centroid variant
python tennis-stroke-detection/Thesis/slice_csv_by_llc.py --variant shoulder_norm
"""

import argparse, json, re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ───────── project paths ─────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()                 # …/Thesis/slice_csv_by_llc.py
ROOT      = THIS_FILE.parents[1]                     # tennis-stroke-detection/
UNPROC    = ROOT / "unprocessed_videos"              # holds *.llc files

VARIANT_MAP = {
    "hip_norm":        "Hip-centered",
    "shoulder_norm":   "Shoulder-centroid",
    "torso_norm":      "Torso-centroid",
    "procrustes_norm": "Procrustes",
}

# ───────── helpers ──────────────────────────────────────────────────
def load_segments(llc_path: Path):
    """
    Robustly parse a LosslessCut .llc file, even when it uses
       • bare keys            version: 1,
       • single quotes        'video.mp4'
       • trailing commas      },   ],   },
    Returns  [(start_sec, end_sec), …]  as floats.
    """
    txt = llc_path.read_text()

    # 1) quote bare keys   { key: → { "key":
    txt = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', txt)

    # 2) convert single→double quotes
    txt = txt.replace("'", '"')

    # 3) drop trailing commas before } or ]
    txt = re.sub(r',\s*([}\]])', r'\1', txt)

    data = json.loads(txt)

    segs = []
    for seg in data.get("cutSegments", []):
        try:
            segs.append((float(seg["start"]), float(seg["end"])))
        except (KeyError, ValueError, TypeError):
            # skip malformed segment but continue processing file
            continue
    return segs

def slice_csv_one(csv_path: Path, llc_path: Path, clips_dir: Path, fps: int = 30):
    """Cut one normalised CSV into per-stroke clips based on llc_path."""
    df = pd.read_csv(csv_path)

    for idx, (t0, t1) in enumerate(load_segments(llc_path), 1):
        f0, f1 = int(t0 * fps), int(t1 * fps)
        clip   = df[(df.frame_index >= f0) & (df.frame_index <= f1)].copy()
        if clip.empty:
            continue
        clip.frame_index -= f0

        clips_dir.mkdir(parents=True, exist_ok=True)
        # video_1_data_normalised → video_1_clip_1_<variant>.csv
        base_name = re.sub(r'_data_normalised$', '', csv_path.stem, flags=re.IGNORECASE)
        out_name  = f"{base_name}_clip_{idx}.csv"
        clip.to_csv(clips_dir / out_name, index=False)

# ───────── main driver ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=list(VARIANT_MAP.keys()))
    args = parser.parse_args()
    variant_key = args.variant
    variant_dir_name = VARIANT_MAP[variant_key]

    # Where the *_normalised.csv files live
    VAR_DIR  = ROOT / "Thesis" / "normalization" / variant_dir_name
    CLIPS_DIR = VAR_DIR / "clips"

    norm_csvs = sorted(VAR_DIR.glob("*_data_normalised.csv"))
    if not norm_csvs:
        print(f"[INFO] No *_data_normalised.csv found in {VAR_DIR}")
        return

    for csv_file in tqdm(norm_csvs, desc=variant_key, unit="file"):
        # video_1_data_normalised → video_1.llc
        base = re.sub(r'_data_normalised$', '', csv_file.stem, flags=re.IGNORECASE)
        llc  = UNPROC / f"{base}.llc"
        if not llc.exists():
            continue
        slice_csv_one(csv_file, llc, CLIPS_DIR)

if __name__ == "__main__":
    main()
