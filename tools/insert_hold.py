#!/usr/bin/env python3
"""Hold the pose at chosen moments, without changing the speed of anything else.

    MUJOCO_GL=egl .venv/bin/python tools/insert_hold.py in.csv out.csv \
        --hold 269,1.0 --hold 328,1.0

A fighter sets before a strike. T11 runs the pull-back straight into the crouch and
the crouch straight into the swing, so the beats read as one continuous slide. This
adds a hold at the end of the pull-back and another at the bottom of the crouch.

The holds are NOT duplicated frames. Repeating a frame stops the motion dead, and a
step from full speed to zero is an infinite acceleration for anything tracking
velocity — which SONIC's reward does, through body_linvel and body_angvel. Instead
the clip is re-timed: a speed profile that eases to zero at the hold, stays there,
and eases back, integrated to give the frame each output frame reads from. Position,
velocity and acceleration all stay continuous.

Joint angles lerp between source frames; the root rotation slerps, because
interpolating an Euler triple tears the orientation apart wherever one wraps.
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
    ap.add_argument("--hold", action="append", required=True,
                    metavar="FRAME,SECONDS", help="repeatable")
    ap.add_argument("--ease", type=float, default=0.20,
                    help="seconds to ease in and out of each hold")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    jc = joint_cols(df)
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]
    T = len(df)
    holds = []
    for h in args.hold:
        f, s = h.split(",")
        holds.append((int(f), float(s)))
    holds.sort()

    ease = max(1, int(args.ease * args.fps))
    total_extra = sum(int(round(s * args.fps)) for _, s in holds)
    n_out = T + total_extra

    # Speed profile over the OUTPUT timeline: 1 normally, 0 through each hold, with a
    # smoothstep ramp either side so nothing steps.
    speed = np.ones(n_out)
    offset = 0
    for f, s in holds:
        n_hold = int(round(s * args.fps))
        centre = f + offset
        a, b = centre, centre + n_hold
        speed[a:b] = 0.0
        for k in range(ease):
            w = k / ease
            u = w * w * (3 - 2 * w)                 # smoothstep
            if a - ease + k >= 0:
                speed[a - ease + k] = min(speed[a - ease + k], 1.0 - u)
            if b + k < n_out:
                speed[b + k] = max(speed[b + k], u)
        offset += n_hold

    # integrate to the source frame each output frame reads from
    u = np.concatenate([[0.0], np.cumsum(speed)[:-1]])
    u = u / u[-1] * (T - 1)

    out = pd.DataFrame({c: np.interp(u, np.arange(T), df[c].values)
                        for c in df.columns if c not in rc})
    rot = Rotation.from_euler("xyz", df[rc].values, degrees=True)
    eul = Slerp(np.arange(T), rot)(np.clip(u, 0, T - 1)).as_euler("xyz", degrees=True)
    for i, c in enumerate(rc):
        out[c] = eul[:, i]
    out = out[list(df.columns)]
    if "Frame" in out.columns:
        out["Frame"] = np.arange(n_out)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")

    q = np.deg2rad(out[jc].values)
    acc = float(np.abs(np.diff(q, n=2, axis=0) * args.fps ** 2).mean())
    print(f"{os.path.basename(args.in_csv)} -> {os.path.basename(args.out_csv)}")
    for f, s in holds:
        print(f"  hold {s:.1f}s at frame {f} ({f / args.fps:.2f}s)")
    print(f"  {T} -> {n_out} frames   {T / args.fps:.2f} -> {n_out / args.fps:.2f} s")
    print(f"  mean |joint accel| {acc:.1f} rad/s^2   (BONES-SEED range 0.9-8.2)")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
