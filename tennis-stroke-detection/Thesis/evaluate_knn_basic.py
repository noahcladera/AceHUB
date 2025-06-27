#!/usr/bin/env python3
"""
eval_knn_auto.py  –  precision@5 and nDCG@5
using the “same-source-video = relevant” rule.
"""

import re, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[2]          # tennis-stroke-detection/
VAR_DIR = {
    "hip_norm":        ROOT/"Thesis/normalization/Hip-centered",
    "shoulder_norm":   ROOT/"Thesis/normalization/Shoulder-centroid",
    "torso_norm":      ROOT/"Thesis/normalization/Torso-centroid",
    "procrustes_norm": ROOT/"Thesis/normalization/Procrustes",
}

def same_source(a: str, b: str) -> bool:
    """video_XX prefix match ⇒ ‘relevant’."""
    return re.match(r"(video_\d+)_", a).group(1) == re.match(r"(video_\d+)_", b).group(1)

def load_test_vectors(vdir: Path):
    X   = np.load(next((vdir/"data/Test").glob("*_vectors.npy")))
    tst = sorted((vdir/"data/Test").glob("*_timed.csv"))
    return X, [p.name for p in tst]

def evaluate_variant(vname: str, vdir: Path, k: int = 5):
    knn   = pickle.load(open(next(vdir.glob("*_knn.pkl")), "rb"))
    lut   = json.load(open(next(vdir.glob("*_train_lookup.json"))))
    X, test_clips = load_test_vectors(vdir)

    precisions, ndcgs = [], []
    for i, qname in enumerate(tqdm(test_clips, desc=vname)):
        dist, idx = knn.kneighbors(X[i].reshape(1,-1), n_neighbors=k)
        neigh = [lut[str(j)] for j in idx[0]]

        rel   = np.array([same_source(qname, n) for n in neigh], int)
        precisions.append(rel.mean())

        # nDCG expects “relevance scores”; we use binary relevance.
        ndcgs.append(
            ndcg_score(rel.reshape(1,-1), 1/(dist+1e-9))
        )
    return np.mean(precisions), np.mean(ndcgs)

# ---------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VAR_DIR.keys())
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-k", type=int, default=5, help="neighbours (default 5)")
    args = ap.parse_args()

    todo = VAR_DIR if args.all else {args.variant: VAR_DIR[args.variant]}
    print(f"{'variant':15}  prec@{args.k}   nDCG@{args.k}")
    print("-"*36)
    for name, vdir in todo.items():
        p, n = evaluate_variant(name, vdir, k=args.k)
        print(f"{name:15}  {p:6.3f}     {n:7.3f}")
