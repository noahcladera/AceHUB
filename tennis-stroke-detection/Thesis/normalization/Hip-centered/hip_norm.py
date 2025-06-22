#!/usr/bin/env python3
"""
hip_norm.py — hip–centering + torso-scale normalisation
=======================================================

Single file
-----------
    python tennis-stroke-detection/Thesis/normalization/Hip-centered/hip_norm.py \
           tennis-stroke-detection/Thesis/video_csv/video_1_data.csv

Batch all full-length CSVs in Thesis/video_csv/
----------------------------------------------
    python tennis-stroke-detection/Thesis/normalization/Hip-centered/hip_norm.py
    # or add --batch explicitly
"""

import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Project paths  (auto-resolve no matter where you launch the script)
# File is at  …/Thesis/normalization/Hip-centered/hip_norm.py
# parents[0] → Hip-centered
# parents[1] → normalization
# parents[2] → Thesis
# parents[3] → tennis-stroke-detection   ← repo root we need
# ──────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]                    # tennis-stroke-detection/
RAW_DIR   = REPO_ROOT / "Thesis" / "video_csv"      # input full-video CSVs
OUT_DIR   = THIS_FILE.parent                        # …/Hip-centered/
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
HIP_IDS       = [23, 24]            # left & right hip
SHOULDER_IDS  = [11, 12]            # left & right shoulder
AXES          = ("x", "y", "z")

def hip_center_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Return hip-centred, torso-scaled dataframe (landmark columns only)."""
    out = df.copy()

    # (frames, 33, 3)
    cols_xyz = [f"lm_{i}_{ax}" for i in range(33) for ax in AXES]
    coords   = df[cols_xyz].values.reshape(len(df), 33, 3)

    P = coords[:, HIP_IDS].mean(axis=1, keepdims=True)        # pelvis mid-point
    S = coords[:, SHOULDER_IDS].mean(axis=1, keepdims=True)   # shoulder mid-point
    scale = np.linalg.norm(S - P, axis=2, keepdims=True)      # torso length
    scale[scale == 0] = 1.0

    norm = (coords - P) / scale
    out[cols_xyz] = norm.reshape(len(df), -1)
    return out

# ──────────────────────────────────────────────────────────────────────
def normalise_file(in_path: Path):
    if not in_path.is_file():
        print(f"[SKIP] {in_path.name} not found")
        return

    df_norm  = hip_center_normalise(pd.read_csv(in_path))
    out_name = in_path.stem + "_normalised.csv"               # video_1_data_normalised.csv
    out_path = OUT_DIR / out_name
    df_norm.to_csv(out_path, index=False)
    print(f"[✓] {out_path.relative_to(REPO_ROOT)}")

def batch_normalise():
    csv_files = sorted(RAW_DIR.glob("*_data.csv"))
    if not csv_files:
        print(f"[INFO] No *_data.csv found in {RAW_DIR}")
        return
    for fp in tqdm(csv_files, desc="hip-norm", unit="file"):
        normalise_file(fp)

# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="single CSV to normalise")
    ap.add_argument("--batch", action="store_true",
                    help="normalise every *_data.csv in Thesis/video_csv/")
    args = ap.parse_args()

    if args.csv:                         # single-file mode overrides batch
        normalise_file(Path(args.csv).expanduser().resolve())
    else:                                # default or --batch
        batch_normalise()

if __name__ == "__main__":
    main()
