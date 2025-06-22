#!/usr/bin/env python3
import argparse, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from load_vectors import load

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("variant"); ap.add_argument("-k", type=int, default=30)
    a = ap.parse_args()
    X, _ = load(a.variant)
    km = KMeans(n_clusters=a.k, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_)
    db  = davies_bouldin_score(X, km.labels_)
    print(f"{a.variant}: silhouette={sil:.3f}  DB={db:.3f}")
