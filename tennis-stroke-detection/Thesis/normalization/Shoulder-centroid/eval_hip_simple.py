#!/usr/bin/env python3
"""
eval_hip_simple.py – precision@5 & nDCG@5 for hip-centred variant
using the neighbours CSV + train-lookup JSON in the *same* folder.
"""

import re, csv, json, math
from pathlib import Path
from collections import defaultdict

HERE      = Path(__file__).resolve().parent
CSV_PATH  = HERE / "shoulder_norm_neighbours.csv"
LUT_PATH  = HERE / "shoulder_norm_train_lookup.json"

K         = 5   # neighbours per query


# ---------- load training-index → filename lookup ---------------------------
with LUT_PATH.open() as f:
    TRAIN_LUT = json.load(f)          # keys are strings ("0", "1", …)

def resolve(name: str) -> str:
    """
    Convert 'TRAIN/708.csv' → actual clip filename via lookup.
    Leaves anything else unchanged.
    """
    if name.startswith("TRAIN/"):
        idx = name.split("/")[1].split(".")[0]   # "708"
        return TRAIN_LUT[idx]
    return name

def same_video(a: str, b: str) -> bool:
    """True if both filenames share the 'video_XX_' prefix."""
    pa = re.match(r"(video_\d+)_", a)
    pb = re.match(r"(video_\d+)_", b)
    return pa and pb and pa.group(1) == pb.group(1)

# ---------------- read neighbour CSV ---------------------------------------
rows = defaultdict(list)          # query -> [(rank, neigh, dist)]
with CSV_PATH.open() as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        neigh = resolve(r["neigh_csv"])
        rows[r["query_csv"]].append(
            (int(r["rank"]), neigh, float(r["euclid_dist"]))
        )

assert all(len(v) == K for v in rows.values()), "each query needs K rows"

# ---------------- per-query metrics ----------------------------------------
precisions, ndcgs = [], []

for q, neigh_list in rows.items():
    neigh_list.sort(key=lambda t: t[0])        # by rank 1…K
    rel = [1 if same_video(q, n) else 0 for _, n, _ in neigh_list]

    # precision@K
    precisions.append(sum(rel) / K)

    # nDCG@K  (binary relevance, log2 discount)
    dcg   = sum(rel[i] / math.log2(i+2) for i in range(K))
    ideal = sum(sorted(rel, reverse=True)[i] / math.log2(i+2) for i in range(K))
    ndcgs.append(dcg / ideal if ideal else 0.0)

print(f"Queries evaluated : {len(rows):>4}")
print(f"precision@{K}      : {sum(precisions)/len(precisions):.3f}")
print(f"nDCG@{K}           : {sum(ndcgs)/len(ndcgs):.3f}")
