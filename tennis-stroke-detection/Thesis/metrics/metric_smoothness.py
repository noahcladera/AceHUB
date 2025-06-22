#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from load_vectors import VAR  # dict variant → dir

def smoothness(csv_path):
    df = pd.read_csv(csv_path, usecols=lambda c: c.startswith("lm_"))
    arr = df.values.reshape(len(df), 33, 3)
    v   = np.linalg.norm(arr[1:] - arr[:-1], axis=2).mean(axis=1)
    return v.mean()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("variant")
    a = ap.parse_args()
    clip_dir = VAR[a.variant] / "data/Training"        # any split ok
    vals = [smoothness(fp) for fp in tqdm(clip_dir.glob("*.csv"), unit="clip")]
    print(f"{a.variant}: mean Δframe = {np.mean(vals):.5f}")
