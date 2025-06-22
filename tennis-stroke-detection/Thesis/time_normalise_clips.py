#!/usr/bin/env python3
"""
time_normalise_clips.py
-----------------------
Resample every *_clip_*.csv to 120 frames (linear interpolation).

Examples
--------
# Normalise hip-centred clips only
python time_normalise_clips.py --variant hip_norm

# Normalise ALL four variants
python time_normalise_clips.py --all
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

NFR = 120  # target frame count

# ----------------------------------------------------------------------
# Locate project directories  (this file lives in  .../tennis-stroke-detection/Thesis/)
ROOT = Path(__file__).resolve().parents[1]        # ← repo root = tennis-stroke-detection
NORM = ROOT / "Thesis" / "normalization"

DIR_MAP = {
    "hip_norm":        NORM / "Hip-centered"      / "clips",
    "shoulder_norm":   NORM / "Shoulder-centroid" / "clips",
    "torso_norm":      NORM / "Torso-centroid"    / "clips",
    "procrustes_norm": NORM / "Procrustes"        / "clips",
}

# ----------------------------------------------------------------------
def interp_df(df: pd.DataFrame, n=NFR) -> pd.DataFrame:
    """Return *df* resampled to *n* rows via linear interpolation."""
    orig_i   = np.linspace(0, 1, len(df))
    target_i = np.linspace(0, 1, n)
    data = {col: np.interp(target_i, orig_i, df[col].values)
            for col in df.columns}
    out = pd.DataFrame(data)
    out["frame_index"] = np.arange(n)       # tidy index
    return out

def process_dir(clip_dir: Path):
    csvs = sorted(clip_dir.glob("*_clip_*.csv"))
    if not csvs:
        print(f"[WARN] No clips found in {clip_dir}")
        return
    for fp in tqdm(csvs, desc=clip_dir.name, unit="clip"):
        out_fp = fp.with_name(fp.stem + "_timed.csv")
        if out_fp.exists():
            continue
        df_timed = interp_df(pd.read_csv(fp))
        df_timed.to_csv(out_fp, index=False)

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=DIR_MAP.keys(),
                    help="normalise clips for ONE variant")
    ap.add_argument("--all", action="store_true",
                    help="normalise clips for ALL variants")
    args = ap.parse_args()

    if args.all:
        targets = DIR_MAP.values()
    elif args.variant:
        targets = [DIR_MAP[args.variant]]
    else:
        ap.print_help(); return

    for d in targets:
        process_dir(d)

if __name__ == "__main__":
    main()
