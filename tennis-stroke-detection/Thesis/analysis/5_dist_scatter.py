#!/usr/bin/env python3
"""
5_dist_vs_score.py  –  sanity-check “distance ↔ coach score”
-------------------------------------------------------------
* needs  clean_ratings.pkl   (Queryvideo, Recommended, Variant, Eval_num)
* needs  *_neighbours.csv    with Euclid_dist & rank columns
      – we keep only the rank-1 neighbour for each query.
"""

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import re

# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]        # tennis-stroke-detection/

# ----- 1) coach ratings ------------------------------------------------
ratings_path = ROOT / "Thesis/analysis/clean_ratings.pkl"
ratings      = pickle.load(open(ratings_path, "rb"))
print(f"Ratings loaded : {len(ratings)} rows")

# helper: strip any trailing “_timed” and the suffix (.csv / .mp4)
def clip_id(name):
    if not isinstance(name, str):
        return None                      # ← skip non-string entries
    stem = re.sub(r'\.(mp4|csv)$', '', name, flags=re.I)
    return re.sub(r'_timed$', '', stem, flags=re.I)

ratings["ClipID"] = ratings["Queryvideo"].map(clip_id)
ratings = ratings.dropna(subset=["ClipID"])

# ----- 2) neighbour distances (rank-1 only) ----------------------------
csv_files = {
    "hip"       : ROOT / "Thesis/normalization/Hip-centered/hip_norm_neighbours.csv",
    "shoulder"  : ROOT / "Thesis/normalization/Shoulder-centroid/shoulder_norm_neighbours.csv",
    "torso"     : ROOT / "Thesis/normalization/Torso-centroid/torso_norm_neighbours.csv",
    "procrustes": ROOT / "Thesis/normalization/Procrustes/procrustes_norm_neighbours.csv",
}

frames = []
for var, fp in csv_files.items():
    df = pd.read_csv(fp)

    # --- harmonise column names – tolerant to your earlier variants ----
    df = df.rename(columns={
        next(c for c in df if c.lower().startswith("query")) : "Queryvideo",
        next(c for c in df if c.lower().startswith(("neigh","recom"))) : "Recommended",
        next(c for c in df if c.lower().startswith("euclid")) : "Euclid_dist"
    })

    # keep only top suggestion
    if "rank" in df.columns:
        df = df.query("rank == 1").copy()

    df["Variant"] = var
    df["ClipID"]  = df["Queryvideo"].map(clip_id)
    frames.append(df[["ClipID", "Euclid_dist", "Variant"]])

distances = pd.concat(frames, ignore_index=True)
print(f"Distances      : {len(distances)} rank-1 rows")

# ----- 3) merge on ClipID + Variant -----------------------------------
merged = ratings.merge(distances, on=["ClipID", "Variant"], how="inner")
print(f"Merged dataset : {len(merged)} rows")

# ----- 4) correlation --------------------------------------------------
rho, p = spearmanr(merged["Euclid_dist"], merged["Eval_num"])
print(f"Spearman ρ = {rho:.3f}   (p = {p:.4g})")

# ----- 5) scatter plot -------------------------------------------------
sns.set(style="whitegrid")
ax = sns.scatterplot(data=merged,
                     x="Euclid_dist", y="Eval_num",
                     hue="Variant", s=35, alpha=0.75)
ax.invert_xaxis()
ax.set_ylabel("Coach score  (yes = 1,  maybe = 0.5,  no = 0)")
ax.set_xlabel("Euclidean distance to query clip")
plt.title("Rank-1 neighbour distance vs. coach score")
plt.tight_layout()

out_png = ROOT / "Thesis/analysis/dist_vs_score.png"
plt.savefig(out_png, dpi=140)
print(f"[✓] plot saved → {out_png.relative_to(ROOT)}")
