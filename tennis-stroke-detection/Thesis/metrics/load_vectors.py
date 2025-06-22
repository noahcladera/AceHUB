#!/usr/bin/env python3
"""
load_vectors.py  –  give me variant name -> (X, clips)
X :  numpy array  (#clips, 11880)  in same order as clips list
"""
import numpy as np, pickle, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # tennis-stroke-detection/
VAR = {
    "hip_norm":        ROOT/"normalization/Hip-centered",
    "shoulder_norm":   ROOT/"normalization/Shoulder-centroid",
    "torso_norm":      ROOT/"normalization/Torso-centroid",
    "procrustes_norm": ROOT/"normalization/Procrustes",
}

def load(variant):
    vdir = VAR[variant]
    X = np.load(vdir/f"{variant}_vectors.npy")
    clips = pickle.load(open(vdir/"clips.pkl", "rb"))
    return X, clips
