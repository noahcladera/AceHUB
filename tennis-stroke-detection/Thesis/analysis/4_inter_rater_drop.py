#!/usr/bin/env python3
"""
4_inter_rater_drop.py  –– Fleiss κ with “absent‐vote” padding
-------------------------------------------------------------
* Looks for clean_ratings.pkl **in the same folder as this script**.
* Handles missing votes by padding a 3rd “abstain” category.
"""

from pathlib import Path
import pickle, pandas as pd, numpy as np, warnings
from statsmodels.stats.inter_rater import fleiss_kappa

HERE = Path(__file__).parent                   # …/analysis
PKL  = HERE / "clean_ratings.pkl"              # <— no more duplicated path

# ------------------------------------------------------------------
print("Loading", PKL)
R = pickle.load(open(PKL, "rb"))

if "CoachID" not in R.columns:
    warnings.warn("Only one coach – Fleiss κ undefined"); quit()

# keep only definite yes / no
R = R[R.Eval_num.isin([0, 1])]
n_raters = R.CoachID.nunique()

# counts matrix with a 3rd “absent” column (code 2)
cnt = (R.pivot_table(index="Queryvideo",
                     columns="Eval_num",
                     values="CoachID",
                     aggfunc="count",
                     fill_value=0)
         .reindex(columns=[0, 1, 2], fill_value=0))      # 0=no, 1=yes, 2=absent
cnt[2] = n_raters - cnt[[0, 1]].sum(axis=1)              # pad missing votes

matrix = cnt.to_numpy(int)
κ = fleiss_kappa(matrix)

print(f"Included clips : {matrix.shape[0]}")
print(f"Raters         : {n_raters}")
print(f"Padded ‘absent’: {matrix[:,2].sum()}")
print(f"Fleiss κ       : {κ:.3f}")
