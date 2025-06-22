#!/usr/bin/env python3
"""
build_knn_split.py  —  fit k-NN indices using the new Training / Validation / Test
folder structure.

Usage examples
--------------
# hip-centred only, k=5, no PCA
python build_knn_split.py --variant hip_norm

# all four variants, k=7, keep 100-D PCA
python build_knn_split.py --all --k 7 --pca 100
"""

import argparse, random, pickle, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]              # tennis-stroke-detection/
N_FR  = 120;  LM = 33; AX = 4
RAW_D = N_FR * LM * AX                                  # 15 840

VARIANT_DIR = {
    "hip_norm":        ROOT/"Thesis/normalization/Hip-centered",
    "shoulder_norm":   ROOT/"Thesis/normalization/Shoulder-centroid",
    "torso_norm":      ROOT/"Thesis/normalization/Torso-centroid",
    "procrustes_norm": ROOT/"Thesis/normalization/Procrustes",
}

# ---------- helpers ------------------------------------------------------------
def flatten_csv(fp: Path) -> np.ndarray:
    df  = pd.read_csv(fp, usecols=lambda c: c.startswith("lm_"))
    arr = df.values.flatten()
    if len(arr) != RAW_D:
        raise ValueError(f"{fp.name}: {len(arr)} values (expect {RAW_D})")
    return arr.astype("float32")

def load_split_vectors(variant_path: Path, split: str) -> tuple[np.ndarray, list[Path]]:
    """Return (matrix, clip_paths) for Training / Validation / Test."""
    clips = sorted( (variant_path/"data"/split).glob("*.csv") )
    if not clips:
        raise SystemExit(f"[ERR] no CSVs in {variant_path/'data'/split}")
    X = np.empty((len(clips), RAW_D), dtype="float32")
    for i, fp in enumerate(tqdm(clips, desc=f"{split:<10}", unit="clip")):
        X[i] = flatten_csv(fp)
    return X, clips
# -------------------------------------------------------------------------------

def process_variant(name: str, k: int, pca_dims: int):
    print(f"\n=== {name} ===")
    vdir = VARIANT_DIR[name]

    # ① Training
    X_train, train_clips = load_split_vectors(vdir, "Training")

    # Optional PCA - learn on training, then project everything
    pca = None
    if pca_dims:
        print(f"[INFO] fitting PCA({pca_dims}) on Training")
        pca = PCA(n_components=pca_dims, svd_solver="randomized")
        X_train = pca.fit_transform(X_train)
        pickle.dump(pca, open(vdir/f"{name}_pca.pkl", "wb"))

    # ② Validation / Test
    X_val,  val_clips  = load_split_vectors(vdir, "Validation")
    X_test, test_clips = load_split_vectors(vdir, "Test")
    if pca:                              # project using training PCA
        X_val  = pca.transform(X_val)
        X_test = pca.transform(X_test)

    # ③ Build k-NN (train only)
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    knn.fit(X_train)
    pickle.dump(knn, open(vdir/f"{name}_knn.pkl", "wb"))
    print(f"[✓] {name}: k-NN(k={k}) saved → {vdir.name}/{name}_knn.pkl")

    # ④ Quick sanity check on 3 random validation clips
    print("  sanity-check queries:")
    for q in random.sample(range(len(val_clips)), k=min(3, len(val_clips))):
        dists, idxs = knn.kneighbors([X_val[q]])
        print(f"    query {val_clips[q].name}")
        for rank, (d,i) in enumerate(zip(dists[0], idxs[0]), 1):
            print(f"      #{rank:<2} {train_clips[i].name:<35} d={d:.3f}")
    print()

# -------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_DIR),
                    help="single variant to process")
    ap.add_argument("--all", action="store_true",
                    help="process all four variants")
    ap.add_argument("--k", type=int, default=5,
                    help="k for k-NN (default 5)")
    ap.add_argument("--pca", type=int, default=0,
                    help="dimensionality after PCA (0 = no PCA)")
    args = ap.parse_args()

    targets = VARIANT_DIR.keys() if args.all else [args.variant]
    if not targets or any(t is None for t in targets):
        ap.error("provide --variant <name> OR --all")

    for v in targets:
        process_variant(v, k=args.k, pca_dims=args.pca)

if __name__ == "__main__":
    main()
