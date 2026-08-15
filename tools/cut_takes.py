#!/usr/bin/env python3
"""Cut delivered footage into one file per take, using the pose-derived segments.

    .venv/bin/python tools/cut_takes.py raw/capture-01 data/capture_analysis/verdict.json \
        --out raw/capture-01/takes

Takes come from wrist-speed segmentation in capture_verdict.py, padded either side
so the settle before and after the phrase survives — the brief asked for that
stillness and the retarget uses it to establish a neutral pose.

Downscaled 4K -> 1080p60: the performer is ~55% of frame height, so 1080p still
gives ~600 px of body, far more than any monocular pose model needs, and the files
become workable.
"""

import argparse
import json
import os
import subprocess

PAD = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("verdict")
    ap.add_argument("--out", default=None)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()
    out = args.out or os.path.join(args.indir, "takes")
    os.makedirs(out, exist_ok=True)

    v = json.load(open(args.verdict))
    index = []
    for clip, d in v.items():
        src = os.path.join(args.indir, clip + ".mov")
        if not os.path.exists(src):
            print(f"  !! missing {src}")
            continue
        slug = clip.lower().replace(" ", "_")
        for i, tk in enumerate(d["takes"], 1):
            s = max(0.0, tk["t"][0] - PAD)
            dur = (tk["t"][1] - tk["t"][0]) + 2 * PAD
            dst = os.path.join(out, f"{slug}_take{i:02d}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.3f}",
                 "-i", src, "-t", f"{dur:.3f}", "-an",
                 "-vf", f"scale=-2:{args.height}", "-c:v", "libx264",
                 "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", dst],
                check=True)
            index.append(dict(file=os.path.basename(dst), clip=clip, take=i,
                              src_start=round(s, 2), dur=round(dur, 2),
                              view=d["view"],
                              deepest_crouch=tk["deepest_crouch"],
                              max_reach=tk["max_reach"],
                              wrist_above_head=tk["wrist_above_head"],
                              peak_wrist_speed=tk["peak_wrist_speed"]))
            print(f"  {os.path.basename(dst)}  {dur:.1f}s")

    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"\n{len(index)} takes -> {out}")


if __name__ == "__main__":
    main()
