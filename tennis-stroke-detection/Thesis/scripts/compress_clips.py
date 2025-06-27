#!/usr/bin/env python3
"""
compress_clips.py
-----------------
Recursively walk tennis-stroke-detection/clips_raw/,
write a compressed copy of every *.mp4 to
tennis-stroke-detection/Thesis/compressed_clips/
keeping the same relative path & file-name.

Usage
-----
python compress_clips.py               # defaults: width 640, CRF 28
python compress_clips.py --width 480   # change long-side
python compress_clips.py --crf 24      # higher quality / larger file
"""

import subprocess, argparse
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[2]   # ← parents[2] instead of [1]
     # tennis-stroke-detection/
SRC_ROOT  = ROOT / "clips_raw"
DST_ROOT  = ROOT / "Thesis" / "compressed_clips"
DST_ROOT.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------------

def compress(src: Path, dst: Path, width: int, crf: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():            # skip already done
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale='min({width},iw)':-2",      # shrink only if wider than <width>
        "-c:v", "libx264", "-preset", "slow",
        "-crf", str(crf),
        "-an",                                     # strip audio (save MBs)
        str(dst)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640, help="max width (px)")
    ap.add_argument("--crf",  type=int, default=28,  help="ffmpeg CRF (smaller = better quality, bigger = smaller files)")
    args = ap.parse_args()

    clips = [p for p in SRC_ROOT.rglob("*.mp4")]
    if not clips:
        raise SystemExit(f"[ERR] no mp4 files found under {SRC_ROOT}")

    print(f"Compressing {len(clips)} clips  →  {DST_ROOT}")
    for src in tqdm(clips, unit="clip"):
        rel  = src.relative_to(SRC_ROOT)           # keep sub-folders if any
        dst  = DST_ROOT / rel
        compress(src, dst, width=args.width, crf=args.crf)

    print("[✓] done")

if __name__ == "__main__":
    main()
