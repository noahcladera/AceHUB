#!/usr/bin/env python3
"""
evaluate_knn_basic.py – precision@5 & nDCG@5 using “same source video” rule.

Run:
  python evaluate_knn_basic.py --all
"""
import re, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[1]
VARIANT_DIRS = {
    "hip_norm":        ROOT/"Thesis/normalization/Hip-centered",
    "shoulder_norm":   ROOT/"Thesis/normalization/Shoulder-centroid",
    "torso_norm":      ROOT/"Thesis/normalization/Torso-centroid",
    "procrustes_norm": ROOT/"Thesis/normalization/Procrustes",
}

def same_video(a, b):
    return re.match(r"(video_\d+)_", a).group(1) == re.match(r"(video_\d+)_", b).group(1)

def load_vectors(split_dir):
    X = np.load(next(split_dir.glob("*_vectors.npy")))
    clips = [p.name for p in sorted(split_dir.glob("*.csv"))]
    return X, clips

def evaluate(variant_dir, k=5):
    knn   = pickle.load(open(next(variant_dir.glob("*_knn.pkl")), "rb"))
    X, clips = load_vectors(variant_dir / "data" / "Test")
    prec, ndcg = [], []
    for i, q in enumerate(tqdm(clips, desc=variant_dir.name)):
        dist, idx = knn.kneighbors(X[i].reshape(1, -1), n_neighbors=k)
        neigh = [clips[j] for j in idx[0]]
        rel   = np.array([same_video(q,n) for n in neigh], int)
        prec.append(rel.mean())
        ndcg.append(ndcg_score(rel.reshape(1,-1), 1/(dist+1e-9)))
    return np.mean(prec), np.mean(ndcg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANT_DIRS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    variants = VARIANT_DIRS if args.all else {args.variant: VARIANT_DIRS[args.variant]}
    print(f"{'variant':15}  prec@5   nDCG@5")
    print("-"*34)
    for name, vdir in variants.items():
        p, n = evaluate(vdir)
        print(f"{name:15}  {p:6.3f}   {n:7.3f}")

if __name__ == "__main__":
    main()
