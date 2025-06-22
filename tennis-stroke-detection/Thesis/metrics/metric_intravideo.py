#!/usr/bin/env python3
import re, numpy as np, argparse
from sklearn.neighbors import NearestNeighbors
from load_vectors import load

vid_id = lambda s: re.search(r"video_(\d+)_", s).group(1)

def p_at5_video(X, clips):
    knn = NearestNeighbors(n_neighbors=6).fit(X)
    hits = []
    for i, q in enumerate(X):
        d, idx = knn.kneighbors(q.reshape(1,-1), n_neighbors=6)
        neigh = [clips[j] for j in idx[0][1:]]
        hits.append( sum(vid_id(n)==vid_id(clips[i]) for n in neigh) / 5 )
    return np.mean(hits)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("variant")
    args = ap.parse_args()
    X, clips = load(args.variant)
    print(f"{args.variant}: P@5_video = {p_at5_video(X, clips):.3f}")