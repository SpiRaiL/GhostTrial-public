#!/usr/bin/env python3
"""Put the reconstructed feet flat on the floor while they are in contact.

    .venv/bin/python tools/flatten_feet.py --src in.bvh --out out.bvh --amount 0.9

The right ankle roll on the G1 is pinned at its stop for ~84% of the clip no
matter how the stance is changed, and it never comes below +8.3 deg even in the
idle frames. The reason is in the source: measuring the reconstruction, neither
sole is ever flat — about 30 deg off horizontal on the left and 42 on the right,
with the right much more variable. A foot that is never flat forces the ankle to
carry that error permanently, and no stance work can fix it.

So each foot is rotated, while it is on the ground, by the minimal rotation that
brings its sole normal to vertical. The correction fades out as the foot lifts, so
genuine foot-lift is untouched.

The sole-normal axis is not assumed — it is found by testing all six local axes of
the foot bone and taking whichever points most nearly upward across the clip
(local -y on the left, +y on the right, which is the mirrored skeleton convention).
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--amount", type=float, default=0.9,
                help="fraction of the tilt to remove while the foot is grounded")
ap.add_argument("--lift", type=float, default=0.06,
                help="metres above the floor at which the correction has faded out")
args = ap.parse_args()

names, off, par, channels, data, T, fps = parse_bvh(args.src)
idx = {n: i for i, n in enumerate(names)}
pos, rots = fk(names, off, par, channels, data, return_rots=True)
UP = np.array([0.0, 1.0, 0.0])

col, cols = 0, {}
for j, n in enumerate(names):
    cols[n] = col
    col += len(channels[j])

feet_i = [idx["LeftToeBase"], idx["RightToeBase"]]
floor = float(np.percentile(pos[:, feet_i, 1].min(axis=1), 2))
out = data.copy()
AXES = {}

for side in ("Left", "Right"):
    bone = side + "Foot"
    ks = [k for k, c in enumerate(channels[idx[bone]]) if c.endswith("rotation")]
    seq = "".join(channels[idx[bone]][k][0] for k in ks)
    rc = [cols[bone] + k for k in ks]
    R = rots[:, idx[bone]]

    # The sole normal must be PERPENDICULAR to the foot's long axis. Picking the
    # "most upward" local axis alone is not enough — it changes with the pose, and
    # after the stance edits it selected a different axis on the same skeleton.
    long_ax = pos[:, idx[side + "ToeBase"]] - pos[:, idx[side + "Foot"]]
    long_ax /= np.linalg.norm(long_ax, axis=1, keepdims=True) + 1e-12
    best = None
    for ax in range(3):
        for sgn in (1, -1):
            v = sgn * R[:, :, ax]
            perp = 1.0 - np.abs(np.einsum("ij,ij->i", v, long_ax)).mean()
            up = np.einsum("ij,j->i", v, UP).mean()
            score = perp * 2.0 + up
            if best is None or score > best[0]:
                best = (score, ax, sgn)
    _, axis_i, axis_s = best

    # contact from the foot's own height range, not an absolute threshold — the
    # floor estimate shifts once the legs have been edited
    h = pos[:, idx[side + "ToeBase"], 1]
    h = h - np.percentile(h, 5)
    contact = np.clip(1.0 - h / args.lift, 0.0, 1.0)

    fixed = 0
    for f in range(T):
        w = contact[f] * args.amount
        if w < 1e-3:
            continue
        n_world = axis_s * R[f, :, axis_i]
        n_world = n_world / (np.linalg.norm(n_world) + 1e-12)
        axis = np.cross(n_world, UP)
        s = np.linalg.norm(axis)
        if s < 1e-6:
            continue
        ang = np.arctan2(s, float(np.dot(n_world, UP)))
        R_world = Rotation.from_rotvec(axis / s * (ang * w)).as_matrix()

        p = par[idx[bone]]
        Rp = rots[f, p]
        new_local = Rp.T @ R_world @ Rp @ (Rp.T @ R[f])
        out[f, rc] = Rotation.from_matrix(new_local).as_euler(seq, degrees=True)
        fixed += 1
    AXES[side] = (axis_i, axis_s)
    print(f"  {side}: sole normal = local {'xyz'[axis_i]}{'+' if axis_s > 0 else '-'}, "
          f"{fixed} of {int((contact > 0.05).sum())} grounded frames corrected")

txt = open(args.src).read()
lines = ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
lines += [" ".join(f"{x:.6f}" for x in r) for r in out]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out, "w").write(txt[:txt.index("MOTION")] + "\n".join(lines) + "\n")

n2, o2, p2, c2, d2, T2, _ = parse_bvh(args.out)
pos2, rots2 = fk(n2, o2, p2, c2, d2, return_rots=True)
def tilt(RR, side, grounded):
    """Mean angle of the sole normal from vertical, over grounded frames only."""
    b = idx[side + "Foot"]
    ax_i, ax_s = AXES[side]
    n = ax_s * RR[:, b, :, ax_i]                  # (T, 3): the sole-normal column
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    a = np.degrees(np.arccos(np.clip(n @ UP, -1, 1)))
    return float(a[grounded].mean())


for side in ("Left", "Right"):
    hh = pos[:, idx[side + "ToeBase"], 1]
    g = (hh - np.percentile(hh, 5)) < args.lift
    print(f"{side} sole tilt from horizontal, grounded frames: "
          f"{tilt(rots, side, g):.1f} -> {tilt(rots2, side, g):.1f} deg")
print(f"wrote {args.out}")
