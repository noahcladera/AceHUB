#!/usr/bin/env python3
"""
eval_knn_auto.py  –  precision@k & nDCG@k for each normalisation variant
using “same-source-video = relevant” as the relevance rule.
"""

import re, json, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import ndcg_score

# ------------------------------------------------------------------ paths -----
ROOT = Path(__file__).resolve().parents[2]         # …/tennis-stroke-detection/
VAR_DIR = {
    "hip_norm":        ROOT/"Thesis/normalization/Hip-centered",
    "shoulder_norm":   ROOT/"Thesis/normalization/Shoulder-centroid",
    "torso_norm":      ROOT/"Thesis/normalization/Torso-centroid",
    "procrustes_norm": ROOT/"Thesis/normalization/Procrustes",
}

# ---------------------------------------------------------------- utils -------
def same_source(a: str, b: str) -> bool:
    """“video_xx” prefix match ⇒ relevant."""
    return re.match(r"(video_\d+)_", a).group(1) == re.match(r"(video_\d+)_", b).group(1)

def load_test_vectors(vdir: Path):
    vec_file = next((vdir/"data/Test").glob("*_vectors.npy"))
    clips    = sorted((vdir/"data/Test").glob("*_timed.csv"))
    X        = np.load(vec_file)
    return X, [p.name for p in clips]

def safe_load(path_glob: Path|str, what: str):
    """Return first match or raise helpful error."""
    if isinstance(path_glob, Path):
        matches = list(path_glob.glob("*"))
    else:                                    # already concrete path str
        matches = [Path(path_glob)] if Path(path_glob).exists() else []
    if not matches:
        raise FileNotFoundError(f"[ERR] cannot find {what}: {path_glob}")
    return matches[0]

def evaluate_variant(vname: str, vdir: Path, k: int = 5):
    knn_pkl  = safe_load(vdir / f"{vname}_knn.pkl",             "k-NN pickle")
    lut_json = safe_load(vdir / f"{vname}_train_lookup.json",   "train-lookup JSON")

    knn   = pickle.load(open(knn_pkl, "rb"))
    lut   = json.load(open(lut_json))
    X, test_clips = load_test_vectors(vdir)

    precisions, ndcgs = [], []
    for i, qname in enumerate(tqdm(test_clips, desc=vname)):
        dist, idx = knn.kneighbors(X[i].reshape(1, -1), n_neighbors=k)
        neigh = [lut[str(j)] for j in idx[0]]

        rel = np.array([same_source(qname, n) for n in neigh], int)
        precisions.append(rel.mean())
        ndcgs.append(ndcg_score(rel.reshape(1,-1), 1/(dist+1e-9)))
    return np.mean(precisions), np.mean(ndcgs)

# ------------------------------------------------------------------ CLI -------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VAR_DIR.keys(), help="single variant")
    ap.add_argument("--all", action="store_true",        help="evaluate all")
    ap.add_argument("-k", type=int, default=5,           help="neighbours (default 5)")
    args = ap.parse_args()

    targets = VAR_DIR if args.all else {args.variant: VAR_DIR[args.variant]}
    print(f"{'variant':15}  prec@{args.k}   nDCG@{args.k}")
    print("-"*36)

    for name, vdir in targets.items():
        p, n = evaluate_variant(name, vdir, k=args.k)
        print(f"{name:15}  {p:6.3f}     {n:7.3f}")
