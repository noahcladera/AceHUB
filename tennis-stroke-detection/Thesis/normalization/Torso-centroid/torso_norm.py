#!/usr/bin/env python3
"""
torso_norm.py — torso-centroid centring + hip–shoulder scale
============================================================

Single file
-----------
    python tennis-stroke-detection/Thesis/normalization/Torso-centroid/torso_norm.py \
           tennis-stroke-detection/Thesis/video_csv/video_1_data.csv

Batch all full-video CSVs
-------------------------
    python tennis-stroke-detection/Thesis/normalization/Torso-centroid/torso_norm.py
"""

import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ───────── project paths ─────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]                       # tennis-stroke-detection/
RAW_DIR   = REPO_ROOT / "Thesis" / "video_csv"
OUT_DIR   = THIS_FILE.parent                           # …/Torso-centroid/
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── landmarks & helpers ───────────────────────────────────────
HIP_IDS       = [23, 24]
SHOULDER_IDS  = [11, 12]
TORSO_IDS     = HIP_IDS + SHOULDER_IDS
AXES          = ("x", "y", "z")

def torso_centroid_normalise(df: pd.DataFrame) -> pd.DataFrame:
    cols_xyz = [f"lm_{i}_{ax}" for i in range(33) for ax in AXES]
    coords   = df[cols_xyz].values.reshape(len(df), 33, 3)

    C = coords[:, TORSO_IDS].mean(axis=1, keepdims=True)      # centre
    P = coords[:, HIP_IDS].mean(axis=1, keepdims=True)
    S = coords[:, SHOULDER_IDS].mean(axis=1, keepdims=True)
    scale = np.linalg.norm(S - P, axis=2, keepdims=True)
    scale[scale == 0] = 1.0

    norm = (coords - C) / scale
    out  = df.copy()
    out[cols_xyz] = norm.reshape(len(df), -1)
    return out

# ───────── per-file / batch drivers ─────────────────────────────────
def normalise_file(csv_in: Path):
    if not csv_in.is_file():
        print(f"[SKIP] {csv_in.name} not found")
        return
    df_norm  = torso_centroid_normalise(pd.read_csv(csv_in))
    out_path = OUT_DIR / f"{csv_in.stem}_normalised.csv"
    df_norm.to_csv(out_path, index=False)
    print(f"[✓] {out_path.relative_to(REPO_ROOT)}")

def batch_normalise():
    csvs = sorted(RAW_DIR.glob("*_data.csv"))
    if not csvs:
        print(f"[INFO] No *_data.csv found in {RAW_DIR}")
        return
    for fp in tqdm(csvs, desc="torso-norm", unit="file"):
        normalise_file(fp)

# ───────── main ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="single CSV to normalise")
    ap.add_argument("--batch", action="store_true")
    args = ap.parse_args()

    if args.csv:
        normalise_file(Path(args.csv).expanduser().resolve())
    else:
        batch_normalise()

if __name__ == "__main__":
    main()
