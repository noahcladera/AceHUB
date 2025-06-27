#!/usr/bin/env python3
"""
00_build_clean_ratings.py
-------------------------
• Loads every “Responses*.csv” sitting one level above /analysis
• Maps yes/maybe/no → 1/.5/0
• Collapses duplicates by (Queryvideo, Variant) with the mean score
• Saves clean_ratings.pkl   (DataFrame with 3 columns)

Run once:
    cd tennis-stroke-detection/Thesis/analysis
    python 00_build_clean_ratings.py
"""

import glob, pickle
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent          # …/Thesis/analysis
CSV_GLOB = ROOT.parent.glob("Responses*.csv")   # any Responses-*.csv

MAP = {"yes": 1.0, "maybe": 0.5, "no": 0.0}

dfs = [pd.read_csv(f) for f in CSV_GLOB]
if not dfs:
    raise SystemExit("No Responses*.csv found!")

raw = pd.concat(dfs, ignore_index=True)

tidy = (raw
        .dropna(subset=["Variant", "Evaluation"])
        .assign(Eval_num=lambda d: d.Evaluation.str.lower().map(MAP))
        .groupby(["Queryvideo", "Variant"], as_index=False)
        .agg(Eval_num=("Eval_num", "mean")))

out = ROOT / "clean_ratings.pkl"
pickle.dump(tidy, open(out, "wb"))
print(f"[✓] wrote {out}  shape={tidy.shape}")
