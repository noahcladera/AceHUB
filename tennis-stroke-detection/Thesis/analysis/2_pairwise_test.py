#!/usr/bin/env python3
"""
2_pairwise_test.py
------------------
Welch’s t-test (unequal N, unequal variance) on the mean coach scores
between every pair of normalisation variants.

Run:
    python 2_pairwise_test.py
"""

import itertools, pickle
from pathlib import Path
from scipy.stats import ttest_ind
import pandas as pd

ROOT = Path(__file__).parent
R = pickle.load(open(ROOT / "clean_ratings.pkl", "rb"))

variants = sorted(R.Variant.unique())
print("Variants:", variants, "\n")

for a, b in itertools.combinations(variants, 2):
    ya = R.loc[R.Variant == a, "Eval_num"]
    yb = R.loc[R.Variant == b, "Eval_num"]
    t, p = ttest_ind(ya, yb, equal_var=False)  # Welch
    print(f"{a:10} vs {b:10}  p = {p:8.4f}   Welch (n={len(ya)}, {len(yb)})")
