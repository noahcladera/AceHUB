#!/usr/bin/env python3
"""
make_train_lookup.py
--------------------
Create a JSON file that maps the *row index in the k-NN object* →
the CSV filename that produced that row.

After it runs you will get, e.g.

    Thesis/normalization/Hip-centered/hip_norm_train_lookup.json
"""

import argparse, json, pickle
from pathlib import Path

# ------------------------------------------------------------------
# resolve repo root no matter where this script lives / is called
REPO = Path(__file__).resolve().parents[2]          # tennis-stroke-detection/
NORM = REPO / "Thesis" / "normalization"

VARIANT_DIR = {
    "hip_norm":        NORM / "Hip-centered",
    "shoulder_norm":   NORM / "Shoulder-centroid",
    "torso_norm":      NORM / "Torso-centroid",
    "procrustes_norm": NORM / "Procrustes",
}
# ------------------------------------------------------------------


def build_lookup(variant: str):
    vdir = VARIANT_DIR[variant]

    # ① load k-NN model -------------------------------------------------
    try:
        knn_path = next(vdir.glob("*_knn.pkl"))
    except StopIteration:
        raise SystemExit(f"[ERR] no *_knn.pkl in {vdir}")

    knn = pickle.load(open(knn_path, "rb"))

    # ② collect training CSV paths -------------------------------------
    train_csvs = sorted((vdir / "data" / "Training").glob("*.csv"))
    if len(train_csvs) != knn._fit_X.shape[0]:
        raise SystemExit(
            f"[ERR] Training clip count ({len(train_csvs)}) ≠ "
            f"knn rows ({knn._fit_X.shape[0]})"
        )

    # ③ write JSON ------------------------------------------------------
    lookup = {i: p.name for i, p in enumerate(train_csvs)}
    out = vdir / f"{variant}_train_lookup.json"
    out.write_text(json.dumps(lookup, indent=2))
    print(f"[✓] wrote {out.relative_to(REPO)}   ({len(lookup)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANT_DIR, required=True,
                    help="hip_norm | shoulder_norm | torso_norm | procrustes_norm")
    args = ap.parse_args()
    build_lookup(args.variant)


if __name__ == "__main__":
    main()
