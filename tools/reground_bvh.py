#!/usr/bin/env python3
"""Put an edited BVH back on the floor.

    .venv/bin/python tools/reground_bvh.py --src in.bvh --out out.bvh

Any edit to the LEG joints moves the feet, not the pelvis, because Hips is the
root of the BVH hierarchy — everything below it swings from there. So straightening
the legs to shallow a crouch lifts the whole figure off the ground, and nothing in
the rest of the chain puts it back: the human BVH ends up airborne, and the
retargeter is then asked to match a floating reference.

That is exactly what happened here. After the stance edits the takes sat 23 cm
(A5) to 41 cm (A7) above the floor, median, with A8 at 38 cm. It is not subtle and
it invalidates everything downstream.

This re-seats the figure by shifting the Hips translation channel so the lowest
foot point rests on the floor. Per frame, because the move has no flight phase —
both feet are meant to be in contact throughout.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["perframe", "mean"], default="perframe")
args = ap.parse_args()

names, off, par, channels, data, T, fps = parse_bvh(args.src)
idx = {n: i for i, n in enumerate(names)}
pos = fk(names, off, par, channels, data)

col, cols = 0, {}
for j, n in enumerate(names):
    cols[n] = col
    col += len(channels[j])

feet = [idx[n] for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")
        if n in idx]
low = pos[:, feet, 1].min(axis=1)               # metres
floor = float(np.percentile(low, 2))
clear = low - floor                              # how far off the ground, per frame
shift = clear if args.mode == "perframe" else np.full(T, clear.mean())

out = data.copy()
out[:, cols["Hips"] + 1] -= shift * 100.0        # BVH channels are centimetres

n2, o2, p2, c2, d2, _, _ = parse_bvh(args.src)
txt = open(args.src).read()
lines = ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
lines += [" ".join(f"{x:.6f}" for x in r) for r in out]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out, "w").write(txt[:txt.index("MOTION")] + "\n".join(lines) + "\n")

n3, o3, p3, c3, d3, T3, _ = parse_bvh(args.out)
pos3 = fk(n3, o3, p3, c3, d3)
low3 = pos3[:, feet, 1].min(axis=1)
h3 = (low3 - np.percentile(low3, 2)) * 100
print(f"foot height above floor: median {np.median(clear) * 100:.1f} -> "
      f"{np.median(h3):.1f} cm   max {clear.max() * 100:.1f} -> {h3.max():.1f} cm")
print(f"wrote {args.out}")
