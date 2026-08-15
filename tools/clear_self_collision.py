#!/usr/bin/env python3
"""Swing the tucked arm clear of the thigh through the crouch.

    .venv/bin/python tools/clear_self_collision.py --deg 12 --out <name>.bvh

Through the deep crouch the performer folds his non-throwing arm in against his
own thigh. On him that is fine — flesh compresses and the camera never sees it.
On the G1 the same pose drives the wrist and elbow links into the hip and knee
links, which WBT-Bench penalises directly, and which sits ~40x above the
self-collision rate of the BONES-SEED motions SONIC was trained on.

So the arm is abducted — swung out away from the body — over the frames that
collide, and only those. Same construction as tools/lower_throw_arm.py: the
rotation is applied to the upper-arm bone in world space and converted back to its
local frame, so the elbow, wrist and hand shape ride along unchanged and only the
arm's clearance from the body changes.

The sign of the rotation is worked out from the motion rather than assumed: a test
rotation is applied both ways and whichever increases the hand-to-hip distance is
the one that abducts.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT = os.path.join(REPO, "data", "human_bvh_edited")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(EDIT, "A2_throw_flat_human.bvh"))
ap.add_argument("--out", default=os.path.join(EDIT, "A3_arm_clear_human.bvh"))
ap.add_argument("--deg", type=float, default=12.0, help="peak abduction, degrees")
ap.add_argument("--start", type=int, default=417, help="first colliding frame (0-based)")
ap.add_argument("--end", type=int, default=515, help="last colliding frame (0-based)")
ap.add_argument("--ramp", type=int, default=25, help="frames to fade in/out over")
ap.add_argument("--side", default="Left")
args = ap.parse_args()

names, off, par, channels, data, T, fps = parse_bvh(args.src)
idx = {n: i for i, n in enumerate(names)}
pos, rots = fk(names, off, par, channels, data, return_rots=True)
UP = np.array([0.0, 1.0, 0.0])

# smooth weight over the colliding span
w = np.zeros(T)
w[args.start:args.end + 1] = 1.0
for k in range(args.ramp):
    f = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / (args.ramp + 1))
    a, b = args.start - args.ramp + k, args.end + args.ramp - k
    if 0 <= a < T:
        w[a] = max(w[a], f)
    if 0 <= b < T:
        w[b] = max(w[b], f)

bone = args.side + "Arm"
hip = args.side + "Leg"
col, cols = 0, {}
for j, n in enumerate(names):
    cols[n] = col
    col += len(channels[j])
rot_k = [k for k, c in enumerate(channels[idx[bone]]) if c.endswith("rotation")]
seq = "".join(channels[idx[bone]][k][0] for k in rot_k)
rot_cols = [cols[bone] + k for k in rot_k]


def axis_at(f):
    """Body-forward in world — rotating about it swings the arm out sideways."""
    lat = pos[f, idx["RightLeg"]] - pos[f, idx["LeftLeg"]]
    lat[1] = 0.0
    if np.linalg.norm(lat) < 1e-6:
        return None
    a = np.cross(UP, lat / np.linalg.norm(lat))
    return a / np.linalg.norm(a)


def rotated_hand(f, ang):
    """Where the hand lands if the upper arm is rotated by `ang` about the axis."""
    a = axis_at(f)
    R = Rotation.from_rotvec(a * np.radians(ang)).as_matrix()
    sh = pos[f, idx[bone]]
    return sh + R @ (pos[f, idx[args.side + "Hand"]] - sh)


# which way is "away from the body"? ask the motion, do not assume
probe = int(np.clip((args.start + args.end) // 2, 0, T - 1))
hip_p = pos[probe, idx[hip]]
d_plus = np.linalg.norm((rotated_hand(probe, +5) - hip_p)[[0, 2]])
d_minus = np.linalg.norm((rotated_hand(probe, -5) - hip_p)[[0, 2]])
sign = 1.0 if d_plus > d_minus else -1.0

out = data.copy()
changed = 0
for f in range(T):
    if w[f] < 1e-3:
        continue
    a = axis_at(f)
    if a is None:
        continue
    R_world = Rotation.from_rotvec(a * np.radians(sign * args.deg * w[f])).as_matrix()
    p = par[idx[bone]]
    Rp = rots[f, p]
    R_local_new = Rp.T @ R_world @ Rp @ (Rp.T @ rots[f, idx[bone]])
    out[f, rot_cols] = Rotation.from_matrix(R_local_new).as_euler(seq, degrees=True)
    changed += 1

txt = open(args.src).read()
lines = ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
lines += [" ".join(f"{x:.6f}" for x in row) for row in out]
open(args.out, "w").write(txt[:txt.index("MOTION")] + "\n".join(lines) + "\n")

n2, o2, p2, c2, d2, _, _ = parse_bvh(args.out)
pos2 = fk(n2, o2, p2, c2, d2)
peak = w > 0.99
before = np.linalg.norm((pos[:, idx[args.side + "Hand"]] - pos[:, idx[hip]])[:, [0, 2]], axis=1)
after = np.linalg.norm((pos2[:, idx[args.side + "Hand"]] - pos2[:, idx[hip]])[:, [0, 2]], axis=1)
print(f"{args.side} arm abducted {args.deg:g} deg (sign {sign:+.0f}) over f{args.start + 1}-"
      f"{args.end + 1} + {args.ramp} ramp, {changed} frames")
print(f"  hand-to-hip clearance over the span: {before[peak].mean() * 100:.1f} -> "
      f"{after[peak].mean() * 100:.1f} cm  (min {before[peak].min() * 100:.1f} -> "
      f"{after[peak].min() * 100:.1f})")
print(f"wrote {args.out}")
