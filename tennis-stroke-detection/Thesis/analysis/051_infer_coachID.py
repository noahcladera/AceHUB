#!/usr/bin/env python3
"""
051_infer_coachID.py  – infer CoachID sessions from timestamp gaps
------------------------------------------------------------------
• Reads a “human responses” CSV exported from your web tool
• Splits rows into sessions whenever the time-gap > GAP_MINUTES
• Saves a tidy DataFrame to clean_ratings.pkl

Usage
-----
    # default: looks for human_responses.csv in the same folder
    python 051_infer_coachID.py

    # or point to a specific file
    python 051_infer_coachID.py  path/to/my_responses.csv
"""

import sys, warnings
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()         # …/Thesis/analysis
DEFAULT_CSV = HERE / "human_responses.csv"     # change if needed
PKL_OUT     = HERE / "clean_ratings.pkl"
GAP_MINUTES = 2                                # session gap threshold
# ------------------------------------------------------------------


def main(csv_path: Path):
    if not csv_path.is_file():
        sys.exit(f"[ERR] CSV not found: {csv_path}")

    print("Loading", csv_path)
    df = pd.read_csv(csv_path)

    # ---------- time column ----------
    # your export column is “Date/time”; adjust if different
    if "Date/time" not in df.columns:
        sys.exit("CSV must contain a ‘Date/time’ column.")
    df["Timestamp"] = pd.to_datetime(df["Date/time"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ---------- session (CoachID) ----------
    gap = df["Timestamp"].diff().gt(pd.Timedelta(minutes=GAP_MINUTES))
    df["CoachID"] = "session_" + gap.cumsum().astype(str).str.zfill(3)

    print("Sessions detected :", df.CoachID.nunique())
    print(df.CoachID.value_counts().head(), "\n")

    # ---------- tidy & recode ----------
    rename = {
        "Queryvideo":  "Queryvideo",
        "Reccomnded":  "Recommended",
        "Variant":     "Variant",
        "Evaluation":  "Eval",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        warnings.warn(f"Missing columns in CSV: {missing}")

    df = df.rename(columns=rename)
    df["Eval_num"] = df["Eval"].str.lower().map(
        {"yes": 1, "no": 0, "maybe": 0.5}
    )

    # ---------- save ----------
    PKL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(PKL_OUT)
    print(f"[✓] Saved to  {PKL_OUT}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    csv_arg = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv_arg)
