#!/usr/bin/env python3
"""
procrustes_norm.py — 3-D Generalised Procrustes normalisation
=============================================================

Single file
-----------
    python tennis-stroke-detection/Thesis/normalization/Procrustes/procrustes_norm.py \
           tennis-stroke-detection/Thesis/video_csv/video_1_data.csv

Batch all full-length CSVs in Thesis/video_csv/
----------------------------------------------
    python tennis-stroke-detection/Thesis/normalization/Procrustes/procrustes_norm.py
"""

import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ───────────────── project paths ─────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]                       # tennis-stroke-detection/
RAW_DIR   = REPO_ROOT / "Thesis" / "video_csv"         # full-video landmark CSVs
OUT_DIR   = THIS_FILE.parent                           # …/Procrustes/
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_FILE = OUT_DIR / "procrustes_template.npy"    # saved once
AXES       = ("x", "y", "z")
LM_COUNT   = 33
HIP_IDS    = [23, 24]
SH_IDS     = [11, 12]

# ───────────────── helper: build or load template ────────────────────
def build_template_from_coords(all_coords: np.ndarray) -> np.ndarray:
    """
    Build a pose template: mean hip-centred shape across all frames
    of all videos (coords  shape=(F,33,3)).
    """
    hip_mid = all_coords[:, HIP_IDS].mean(axis=1, keepdims=True)
    centred = all_coords - hip_mid                               # (F,33,3)
    template = centred.mean(axis=0)                              # (33,3)
    np.save(TEMPLATE_FILE, template)
    print(f"[INFO] Procrustes template saved → {TEMPLATE_FILE.relative_to(REPO_ROOT)}")
    return template

def get_template() -> np.ndarray:
    if TEMPLATE_FILE.exists():
        return np.load(TEMPLATE_FILE)
    # build from all csvs in RAW_DIR (could be slow first time)
    coords_list = []
    for csv in RAW_DIR.glob("*_data.csv"):
        df = pd.read_csv(csv, usecols=[f"lm_{i}_{ax}" for i in range(LM_COUNT) for ax in AXES])
        coords_list.append(df.values.reshape(len(df), LM_COUNT, 3))
    if not coords_list:
        sys.exit("[ERR] No *_data.csv found to build template.")
    all_coords = np.concatenate(coords_list, axis=0)
    return build_template_from_coords(all_coords)

# ───────────────── kabsch alignment ──────────────────────────────────
def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return P aligned (rotated, scaled) on Q using Kabsch in 3-D."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    sP = np.linalg.norm(Pc) / np.sqrt(LM_COUNT) or 1.0
    sQ = np.linalg.norm(Qc) / np.sqrt(LM_COUNT) or 1.0
    Pc /= sP; Qc /= sQ

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:      # reflection fix
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return (Pc @ R) * (sQ / 1.0)

# ───────────────── normalisation routine ─────────────────────────────
def procrustes_normalise(df: pd.DataFrame, template: np.ndarray) -> pd.DataFrame:
    cols_xyz = [f"lm_{i}_{ax}" for i in range(LM_COUNT) for ax in AXES]
    coords   = df[cols_xyz].values.reshape(len(df), LM_COUNT, 3)

    aligned = np.empty_like(coords)
    for f in range(len(df)):
        aligned[f] = kabsch_align(coords[f], template)

    out = df.copy()
    out[cols_xyz] = aligned.reshape(len(df), -1)
    return out

# ───────────────── per-file & batch drivers ──────────────────────────
def normalise_file(csv_in: Path, template: np.ndarray):
    if not csv_in.is_file():
        print(f"[SKIP] {csv_in.name} not found")
        return
    df_norm = procrustes_normalise(pd.read_csv(csv_in), template)
    out_path = OUT_DIR / f"{csv_in.stem}_normalised.csv"
    df_norm.to_csv(out_path, index=False)
    print(f"[✓] {out_path.relative_to(REPO_ROOT)}")

def batch_normalise():
    template = get_template()
    csv_files = sorted(RAW_DIR.glob("*_data.csv"))
    if not csv_files:
        print(f"[INFO] No *_data.csv found in {RAW_DIR}")
        return
    for fp in tqdm(csv_files, desc="procrustes-norm", unit="file"):
        normalise_file(fp, template)

# ─────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="single CSV to normalise")
    ap.add_argument("--batch", action="store_true", help="normalise every *_data.csv")
    args = ap.parse_args()

    if args.csv:
        template = get_template()
        normalise_file(Path(args.csv).expanduser().resolve(), template)
    else:
        batch_normalise()

if __name__ == "__main__":
    main()
