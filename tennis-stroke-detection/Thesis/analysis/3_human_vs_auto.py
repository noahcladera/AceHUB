#!/usr/bin/env python3
"""
3_human_vs_auto.py
==================
Join human-judgement scores (YES / MAYBE / NO) with automatic
precision@5 / nDCG@5 for each normalisation variant.

• Edit the two PATHS below if your files live elsewhere.
• Run:  python 3_human_vs_auto.py
"""

from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# UPDATE THESE TWO PATHS IF NEEDED
ROOT = Path(__file__).parent            # Thesis/analysis/
HUMAN_FILE = ROOT / "human_responses.csv"     # cleaned coaches’ CSV
AUTO_FILE  = ROOT / "auto_metrics.csv"        # built by build_auto_metrics.py
# ------------------------------------------------------------------

# ------------- 1)  Human data -------------------------------------
print("\nLoading human judgements from", HUMAN_FILE)
R = pd.read_csv(HUMAN_FILE)

# expected columns: Variant, Evaluation   (YES / MAYBE / NO …case-insensitive)
R["Variant"] = R["Variant"].str.strip().str.lower()
R["Evaluation"] = R["Evaluation"].str.strip().str.lower()

map_score = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
R = R[R["Evaluation"].isin(map_score)]          # drop any weird rows
R["Score"] = R["Evaluation"].map(map_score)

human = (R.groupby("Variant")["Score"]
           .agg(["mean", "count"])
           .rename(columns={"mean": "Human_mean", "count": "N_human"}))

print("\nHuman aggregates:")
print(human)

# ------------- 2)  Automatic metrics ------------------------------
print("\nLoading automatic metrics from", AUTO_FILE)
auto = (pd.read_csv(AUTO_FILE)
          .assign(Variant=lambda d: d["Variant"].str.strip().str.lower())
          .set_index("Variant"))

print("\nAutomatic metrics:")
print(auto)

# ------------- 3)  Merge & save -----------------------------------
merged = human.join(auto, how="inner")      # inner → only variants present in both
out_path = ROOT / "human_vs_auto.csv"
merged.to_csv(out_path, float_format="%.3f")

print("\n===  HUMAN  vs  AUTO  ===")
print(merged)
print("\n✓ wrote", out_path)
