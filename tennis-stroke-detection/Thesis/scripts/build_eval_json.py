#!/usr/bin/env python3
"""
build_eval_json.py
Convert one variant’s neighbour-CSV  +  train-lookup.json
→  a single  <variant>_eval.json  with structure:
    {
      "query_clip_A.mp4": ["neigh1.mp4", "neigh2.mp4", ...],
      ...
    }
Run:
    python build_eval_json.py hip_norm
"""
import csv, json, argparse
from pathlib import Path

ROOT = Path("tennis-stroke-detection/Thesis/normalization")
DIRMAP = {
    "hip_norm":        "Hip-centered",
    "shoulder_norm":   "Shoulder-centroid",
    "torso_norm":      "Torso-centroid",
    "procrustes_norm": "Procrustes",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=DIRMAP.keys(),
                    help="which normalisation to process")
    args = ap.parse_args()
    vdir = ROOT / DIRMAP[args.variant]

    neigh_csv = vdir / f"{args.variant}_neighbours.csv"
    lut_json  = vdir / f"{args.variant}_train_lookup.json"

    if not neigh_csv.exists() or not lut_json.exists():
        raise SystemExit(f"[ERR] expect {neigh_csv.name} and {lut_json.name}")

    # 1) load train-index → filename lookup
    lookup = json.load(open(lut_json))

    # 2) collect neighbours per query
    rows = {}
    with open(neigh_csv) as f:
        reader = csv.reader(f)
        for q, rank, n_idx, _dist in reader:
            # neighbour indices in the CSV start with TRAIN/xxx
            n_file = lookup.get(n_idx.split('/')[-1], n_idx)
            rows.setdefault(q.replace(".csv", ".mp4"), []).append(
                 n_file.replace(".csv", ".mp4"))

    out = vdir / f"{args.variant}_eval.json"
    json.dump(rows, open(out, "w"), indent=1)
    print("[✓] wrote", out)

if __name__ == "__main__":
    main()
