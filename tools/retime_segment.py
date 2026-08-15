#!/usr/bin/env python3
"""Compress or stretch one stretch of a take, leaving the rest at its own timing.

    MUJOCO_GL=egl .venv/bin/python tools/retime_segment.py in.csv out.csv \
        --range 435,610 --seconds 1.0

The hold between the spear throw and the pull-back runs 175 frames — 2.9 seconds of
real time, which reads as dead air. Only that stretch is resampled; the throw and the
uppercut either side keep the timing they were performed with.

A note on which seconds these are. The CSV is 60 fps, so the take is 16.4 s of real
motion, and that is what the robot performs. The review videos have been rendered at
30 fps, which plays everything at HALF speed and makes the same clip look 32.8 s
long. Frame numbers and --seconds here are real, at 60 fps.

Joint angles lerp — they are independent revolute joints well inside their ranges.
The root rotation does NOT: it goes through quaternion slerp, because interpolating
an Euler triple tears the orientation apart wherever one of them wraps.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_columns import joint_cols  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--range", required=True, help="lo,hi frames to retime")
    ap.add_argument("--seconds", type=float, required=True, help="target length of that stretch")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    jc = joint_cols(df)
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]
    lo, hi = (int(x) for x in args.range.split(","))
    n_new = max(2, int(round(args.seconds * args.fps)))

    rot = Rotation.from_euler("xyz", df[rc].values, degrees=True)
    src = np.arange(lo, hi + 1, dtype=float)
    dst = np.linspace(lo, hi, n_new)

    mid = pd.DataFrame({c: np.interp(dst, src, df[c].values[lo:hi + 1])
                        for c in df.columns if c not in rc})
    eul = Slerp(src, rot[lo:hi + 1])(dst).as_euler("xyz", degrees=True)
    for i, c in enumerate(rc):
        mid[c] = eul[:, i]
    mid = mid[list(df.columns)]

    out = pd.concat([df.iloc[:lo], mid, df.iloc[hi + 1:]], ignore_index=True)
    if "Frame" in out.columns:
        out["Frame"] = np.arange(len(out))
    out.to_csv(args.out_csv, index=False, float_format="%.6f")

    old_s = (hi - lo + 1) / args.fps
    print(f"{os.path.basename(args.in_csv)} -> {os.path.basename(args.out_csv)}")
    print(f"  stretch {lo}..{hi}: {hi - lo + 1} frames ({old_s:.2f} s) -> "
          f"{n_new} frames ({n_new / args.fps:.2f} s)")
    print(f"  clip total {len(df)} -> {len(out)} frames "
          f"({len(df) / args.fps:.2f} -> {len(out) / args.fps:.2f} s real)")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
