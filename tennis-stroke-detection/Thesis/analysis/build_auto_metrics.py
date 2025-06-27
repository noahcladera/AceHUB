#!/usr/bin/env python3
"""
Read the four console‐outputs that you already have hard-coded below and
write auto_metrics.csv  (Variant, prec5, nDCG5).
Edit the numbers once if you re-run the evaluations.
"""

import pandas as pd
from pathlib import Path

# -----------  paste your latest values here  ---------
metrics = [
    # Variant       prec@5   nDCG@5
    ("hip",         0.539,   0.763),
    ("procrustes",  0.574,   0.788),
    ("shoulder",    0.574,   0.748),
    ("torso",       0.554,   0.769),
]
# ----------------------------------------------------

df = pd.DataFrame(metrics, columns=["Variant", "prec5", "nDCG5"])
out = Path(__file__).with_name("auto_metrics.csv")
df.to_csv(out, index=False)
print("✓ wrote", out)
print(df)
