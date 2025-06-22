#!/usr/bin/env python3
"""
Cut every video in unprocessed_videos/ into raw clips using its .llc file.
Clips saved in clips_raw/ as <stem>_clip_<n>.mp4
"""

import subprocess, json, re, sys
from pathlib import Path
from tqdm import tqdm

ROOT    = Path(__file__).resolve().parents[1]          # repo root
RAW_VID = ROOT / "unprocessed_videos"
RAW_CLP = ROOT / "clips_raw"
RAW_CLP.mkdir(exist_ok=True)

def load_segments(llc_path):
    """
    Return list of (startTime, endTime) tuples from a LosslessCut *.llc file.
    Works with the non-strict JSON that LosslessCut writes.
    """
    txt = llc_path.read_text()

    # 1) put quotes around property names   foo: → "foo":
    txt = re.sub(r'([{\[,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', txt)

    # 2) single → double quotes
    txt = txt.replace("'", '"')

    # 3) remove trailing commas  { …,}  or  […,]
    txt = re.sub(r',(\s*[}\]])', r'\1', txt)

    try:
        data = json.loads(txt)
        segs = [(seg["start"], seg["end"]) for seg in data["cutSegments"]]
        return segs
    except Exception as e:                                   # last-ditch fallback
        print(f"[WARN] JSON parse failed on {llc_path.name} ({e}); "
              "using regex fallback")
        pattern = re.compile(r'start\s*"\s*:\s*([\d.]+).*?end\s*"\s*:\s*([\d.]+)',
                             re.S)
        segs = [(float(a), float(b)) for a, b in pattern.findall(txt)]
        if not segs:
            raise RuntimeError(f"No segments found in {llc_path}")
        return segs

def cut(video, llc):
    segs = load_segments(llc)
    for i, (s, e) in enumerate(segs, 1):
        out = RAW_CLP / f"{video.stem}_clip_{i}.mp4"
        if out.exists(): continue
        cmd = ["ffmpeg","-y","-i",str(video),
               "-ss",str(s),"-to",str(e),
               "-c:v","libx264","-preset","fast","-crf","23",
               str(out)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[✓]", out.name)

def main():
    vids = [p for p in RAW_VID.glob("*.*") if p.suffix.lower() in (".mp4",".mov",".avi")]
    for v in vids:
        llc = v.with_suffix(".llc")
        if not llc.exists():
            print(f"[skip] no llc for {v.name}")
            continue
        cut(v, llc)

if __name__ == "__main__":
    main()
