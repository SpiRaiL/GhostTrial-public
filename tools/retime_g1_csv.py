#!/usr/bin/env python3
"""Apply the same swing speed-up to the G1 side that was applied to the human.

    .venv/bin/python tools/retime_g1_csv.py

Otherwise the .blend and the training data disagree: the human motion has the
uppercut at 2x and the robot the performer's original timing, so a policy trained
on it would never produce the move that was actually approved.

The window is taken from the matching human BVH rather than recomputed here — the
G1 retarget has one row per source frame, so the frame numbering is shared, and
deriving it twice from different signals is how the two drift apart.

Column types decide the interpolation. The 29 joint DOFs are independent revolute
angles well inside their ranges, so they lerp safely. The root rotation is an
Euler triple and does not — it goes through quaternion slerp, same as the BVH path.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402
from retime_bvh import SWING_SPEED, RAMP, find_stance, find_swing, rate  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "gemx_g1")
DST = os.path.join(REPO, "data", "gemx_g1_retimed")
BVH = os.path.join(REPO, "data", "human_bvh")

os.makedirs(DST, exist_ok=True)
for fn in sorted(f for f in os.listdir(SRC) if f.endswith(".csv")):
    take = fn[:-4]
    stance, crouch, _, pos, idx, floor = find_stance(
        os.path.join(BVH, f"{take}_human.bvh"))
    a, b, peak = find_swing(pos, idx, floor, crouch)

    head = open(os.path.join(SRC, fn)).readline().strip()
    data = np.loadtxt(os.path.join(SRC, fn), delimiter=",", skiprows=1)
    T = len(data)

    times, t = [], float(stance)
    while t <= T - 1:
        times.append(t)
        t += rate(t, a, b, SWING_SPEED, RAMP)
    times = np.array(times)
    src_f = np.arange(T)

    out = np.zeros((len(times), data.shape[1]))
    out[:, 0] = np.arange(len(times))                        # Frame
    for c in list(range(1, 4)) + list(range(7, data.shape[1])):
        out[:, c] = np.interp(times, src_f, data[:, c])      # root translate + DOFs
    rots = Rotation.from_euler("xyz", data[:, 4:7], degrees=True)
    out[:, 4:7] = Slerp(src_f, rots)(times).as_euler("xyz", degrees=True)

    dst = os.path.join(DST, fn)
    np.savetxt(dst, out, delimiter=",", header=head, comments="", fmt="%.6f")
    print(f"{take}: swing f{a + 1}-{b + 1} at {SWING_SPEED:g}x  ·  "
          f"{T} -> {len(times)} frames  -> {dst}")
