#!/usr/bin/env python3
"""
vectorise_clips.py  —  flatten (and optionally PCA-compress) every
normalised-clip CSV into a single matrix per variant.

Usage examples
--------------
# all four variants, keep 100-D PCA
python vectorise_clips.py --all --pca 100

# hip-centred only, no PCA
python vectorise_clips.py --variant hip_norm
"""

import argparse, pickle, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]          # tennis-stroke-detection/
NFR  = 120                                          # frames/clip after time-norm
LM, AX = 33, 4 
RAW_D  = NFR * LM * AX                              # 11 880

VAR_DIR = {
    "hip_norm":        ROOT/"Thesis/normalization/Hip-centered/clips",
    "shoulder_norm":   ROOT/"Thesis/normalization/Shoulder-centroid/clips",
    "torso_norm":      ROOT/"Thesis/normalization/Torso-centroid/clips",
    "procrustes_norm": ROOT/"Thesis/normalization/Procrustes/clips",
}

def flatten_csv(fp: Path) -> np.ndarray:
    df = pd.read_csv(fp, usecols=lambda c: c.startswith("lm_"))
    arr = df.values.flatten()              # F*33*3
    if len(arr) != RAW_D:                  # guard against stray clip length
        raise ValueError(f"{fp.name}: got {len(arr)} values, expected {RAW_D}")
    return arr.astype("float32")

def build_matrix(clip_dir: Path):
    clips = sorted(clip_dir.glob("*_timed.csv"))
    vecs  = np.empty((len(clips), RAW_D), dtype="float32")
    for i, fp in enumerate(tqdm(clips, desc=clip_dir.parent.name, unit="clip")):
        vecs[i] = flatten_csv(fp)
    return vecs, clips

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VAR_DIR), help="single variant")
    ap.add_argument("--all", action="store_true", help="process all variants")
    ap.add_argument("--pca", type=int, default=0, help="dims after PCA (0 = none)")
    args = ap.parse_args()

    todo = VAR_DIR.keys() if args.all else [args.variant]
    if not todo or any(v is None for v in todo):
        ap.error("Specify --variant <name> or --all")

    for v in todo:
        clip_dir = VAR_DIR[v]
        X_raw, clips = build_matrix(clip_dir)

        if args.pca:
            print(f"[INFO] fitting PCA({args.pca}) on {v}")
            pca = PCA(n_components=args.pca, svd_solver="randomized", whiten=False)
            X = pca.fit_transform(X_raw)
            pickle.dump(pca, open(clip_dir.parent/f"{v}_pca.pkl", "wb"))
        else:
            X = X_raw

        np.save(clip_dir.parent/f"{v}_vectors.npy", X)
        print(f"[✓] {v}: saved {X.shape} to {v}_vectors.npy")

if __name__ == "__main__":
    main()
