#!/usr/bin/env python3
"""
export_neighbours.py
--------------------
Write a CSV with the top-K neighbours (default 5) for every TEST clip,
for each normalisation variant that already has:
    · *_knn.pkl         (fitted on TRAIN)
    · *_splits.npy      (1-D str  array, len = #vectors)
    · *_vectors.npy     (vectors for TRAIN+VAL **only**)
Clip CSVs are read from   normalization/<Variant>/data/Test/*.csv
Output: normalization/<Variant>/<variant>_neighbours.csv
"""

import csv, pickle, re, json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # .../Thesis
N_FR, LM, AX = 120, 33, 4                       # clip length, landmarks, coords
RAW_D = N_FR*LM*AX

VARIANT_DIR = {
    "hip_norm":        ROOT / "normalization" / "Hip-centered",
    "shoulder_norm":   ROOT / "normalization" / "Shoulder-centroid",
    "torso_norm":      ROOT / "normalization" / "Torso-centroid",
    "procrustes_norm": ROOT / "normalization" / "Procrustes",
}
TOP_K = 5
# ----------------------------------------------------------------------


def flatten_csv(fp: Path) -> np.ndarray:
    df = pd.read_csv(fp, usecols=lambda c: c.startswith("lm_"))
    arr = df.values.flatten()
    if len(arr) != RAW_D:
        raise ValueError(f"{fp.name}: {len(arr)} values (expect {RAW_D})")
    return arr.astype("float32")


def process_variant(name: str, vdir: Path):
    print(f"\n=== {name} ===")

    # ---------- load artefacts ------------------------------------------------
    knn: NearestNeighbors = pickle.load(open(vdir / f"{name}_knn.pkl", "rb"))
    splits  = np.load(vdir / f"{name}_splits.npy", allow_pickle=True)
    if splits.ndim == 0:                       # legacy 0-D pickle
        splits = np.asarray(splits.item(), dtype="<U")

    # ---------- gather TEST clips & vectorise ---------------------------------
    test_dir = vdir / "data" / "Test"
    clips = sorted(test_dir.glob("*.csv"))
    if not clips:
        print(f"[SKIP] no CSVs in {test_dir}")
        return

    X_test = np.empty((len(clips), RAW_D), dtype="float32")
    for i, fp in enumerate(tqdm(clips, desc="vectorising", unit="clip")):
        X_test[i] = flatten_csv(fp)

    # ---------- query ---------------------------------------------------------
    out_csv = vdir / f"{name}_neighbours.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["query_csv", "rank", "neigh_csv", "euclid_dist"])
        for q_idx, q_fp in enumerate(tqdm(clips, desc="k-NN", unit="query")):
            dist, idx = knn.kneighbors(X_test[q_idx].reshape(1, -1),
                                       n_neighbors=TOP_K)
            for rank, (d, i) in enumerate(zip(dist[0], idx[0]), 1):
                wr.writerow([q_fp.name, rank, f"TRAIN/{i}.csv", f"{d:.4f}"])

    print(f"[✓] wrote {out_csv.relative_to(ROOT)}")


def main():
    for name, vdir in VARIANT_DIR.items():
        process_variant(name, vdir)


if __name__ == "__main__":
    main()
