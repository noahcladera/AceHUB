#!/usr/bin/env python3
"""
1_variant_stats.py
------------------
Descriptive stats of coach ratings for every normalisation variant:
 • #clips rated
 • mean score       (yes=1 · maybe=.5 · no=0)
 • median
 • standard deviation
 • 95 % confidence interval for the mean

Run:
    cd tennis-stroke-detection/Thesis/analysis
    python 1_variant_stats.py
"""

import math, pickle
from pathlib import Path
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).parent
R = pickle.load(open(ROOT / "clean_ratings.pkl", "rb"))   # built by 00_build_…

rows = []
for v, grp in R.groupby("Variant"):
    n   = len(grp)
    mu  = grp.Eval_num.mean()
    med = grp.Eval_num.median()
    sd  = grp.Eval_num.std(ddof=1)
    se  = sd / math.sqrt(n)
    ci  = t.ppf(0.975, n-1) * se           # 95 % CI
    rows.append([v, n, mu, med, sd, mu-ci, mu+ci])

summary = (pd.DataFrame(rows, columns=[
               "Variant", "N", "Mean", "Median", "SD", "CI_low", "CI_high"])
             .sort_values("Mean", ascending=False))

print("\nPer-variant coach-score summary (yes=1, maybe=.5, no=0)\n")
print(summary.to_string(index=False, float_format="%.3f"))
