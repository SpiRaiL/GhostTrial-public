#!/usr/bin/env python3
"""Take A2 — bring the spear-throwing arm down to horizontal at full extension.

    .venv/bin/python tools/lower_throw_arm.py

In the capture the throwing arm finishes about 13.4 deg above horizontal. Scorpion's
spear goes out flat, so this rotates the whole arm down about a horizontal axis
until the shoulder-to-hand line is level.

The rotation is applied to the UPPER-ARM bone in world space and converted back to
its local frame, so the forearm and hand inherit it: the elbow angle, the wrist and
the hand shape are all preserved exactly, and only the arm's carrying angle changes.
Weighted by how extended the arm is, so it reaches full effect exactly at the throw
and fades to nothing as the arm folds back in — no window boundaries to snap at.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default=os.path.join(REPO, "data", "human_bvh_edited",
                                               "A_side_deepcrouch_human.bvh"))
_ap.add_argument("--out", default=os.path.join(REPO, "data", "human_bvh_edited",
                                               "A2_throw_flat_human.bvh"))
_a = _ap.parse_args()
SRC, DST = _a.src, _a.out

TARGET_ELEV = 0.0     # degrees above horizontal at full extension
EXT_LO, EXT_HI = 0.86, 0.96   # arm extension over which the correction fades in

names, off, par, channels, data, T, fps = parse_bvh(SRC)
idx = {n: i for i, n in enumerate(names)}
pos, rots = fk(names, off, par, channels, data, return_rots=True)

# the throwing arm is the one thrust out while the other stays tucked (same test
# the beat detector uses), searched before the crouch
feet = [idx["LeftToeBase"], idx["RightToeBase"]]
floor = float(np.percentile(pos[:, feet, 1].min(axis=1), 2))
crouch = int(np.argmin(pos[:, idx["Hips"], 1] - floor))


def arm_out(side):
    n = (np.linalg.norm(off[idx[side + "ForeArm"]])
         + np.linalg.norm(off[idx[side + "Hand"]])) * 0.01
    d = pos[:, idx[side + "Hand"]] - pos[:, idx[side + "Arm"]]
    return np.linalg.norm(d[:, [0, 2]], axis=1) / n


gap = arm_out("Right") - arm_out("Left")
f_throw = int(np.argmax(np.abs(gap[:crouch])))
side = "Right" if gap[f_throw] > 0 else "Left"
UP = np.array([0.0, 1.0, 0.0])

sh, hd = pos[:, idx[side + "Arm"]], pos[:, idx[side + "Hand"]]
v = hd - sh
armlen = np.linalg.norm(v, axis=1)
ext = armlen / (armlen.max() + 1e-9)
elev = np.degrees(np.arctan2(v[:, 1], np.linalg.norm(v[:, [0, 2]], axis=1)))

# fade the correction in with extension, and only over the throw half of the clip
w = np.clip((ext - EXT_LO) / (EXT_HI - EXT_LO), 0.0, 1.0)
w[crouch:] = 0.0                       # the uppercut arm is not ours to touch

# Keep only the extension run that actually contains the throw. The arm is also
# near-straight during the WIND-UP, cocked overhead at ~+52 deg, and flattening
# that would delete the very pose that makes the throw read.
on = w > 0
edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
keep = next((r for r in runs if r[0] <= f_throw < r[1]), None)
if keep is None:
    raise SystemExit("no extension run contains the detected throw frame")
mask = np.zeros(T, bool)
mask[keep[0]:keep[1]] = True
w[~mask] = 0.0

# column offsets, and the rotation-channel order for the upper-arm bone
col, cols = 0, {}
for j, n in enumerate(names):
    cols[n] = col
    col += len(channels[j])
bone = side + "Arm"
rot_k = [k for k, c in enumerate(channels[idx[bone]]) if c.endswith("rotation")]
seq = "".join(channels[idx[bone]][k][0] for k in rot_k)
rot_cols = [cols[bone] + k for k in rot_k]

out = data.copy()
changed = 0
for f in range(T):
    # only ever LOWER. The arm is also fully extended during the pull-back that
    # follows the throw, where it already sits a few degrees below horizontal —
    # correcting toward the target there would raise it, which is not the ask.
    if w[f] < 1e-3 or elev[f] <= TARGET_ELEV:
        continue
    delta = np.radians((TARGET_ELEV - elev[f]) * w[f])
    horiz = v[f].copy()
    horiz[1] = 0.0
    if np.linalg.norm(horiz) < 1e-6:
        continue
    # rotate about the horizontal axis perpendicular to the arm's ground track;
    # positive delta raises, negative lowers
    axis = np.cross(UP, horiz / np.linalg.norm(horiz))
    axis /= np.linalg.norm(axis)
    R_world = Rotation.from_rotvec(axis * -delta).as_matrix()

    p = par[idx[bone]]
    Rp = rots[f, p]                              # parent (shoulder) world rotation
    R_local_old = Rp.T @ rots[f, idx[bone]]
    R_local_new = Rp.T @ R_world @ Rp @ R_local_old
    out[f, rot_cols] = Rotation.from_matrix(R_local_new).as_euler(seq, degrees=True)
    changed += 1

txt = open(SRC).read()
hierarchy = txt[:txt.index("MOTION")]
lines = ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
lines += [" ".join(f"{x:.6f}" for x in row) for row in out]
open(DST, "w").write(hierarchy + "\n".join(lines) + "\n")

# verify by re-running FK on what was written
n2, o2, p2, c2, d2, T2, _ = parse_bvh(DST)
pos2 = fk(n2, o2, p2, c2, d2)
v2 = pos2[:, idx[side + "Hand"]] - pos2[:, idx[side + "Arm"]]
elev2 = np.degrees(np.arctan2(v2[:, 1], np.linalg.norm(v2[:, [0, 2]], axis=1)))
hold = (w > 0.9) & (elev > TARGET_ELEV)   # the throw itself, not the pull-back
print(f"throwing arm: {side}   throw at f{f_throw + 1}   {changed} frames adjusted")
print(f"  elevation at full extension: {elev[hold].mean():+.1f} deg -> "
      f"{elev2[hold].mean():+.1f} deg")
print(f"  arm length preserved: {np.linalg.norm(v[hold], axis=1).mean():.4f} m -> "
      f"{np.linalg.norm(v2[hold], axis=1).mean():.4f} m")
print(f"  hand height at the throw: {pos[f_throw, idx[side + 'Hand'], 1]:.3f} m -> "
      f"{pos2[f_throw, idx[side + 'Hand'], 1]:.3f} m")
print(f"wrote {DST}")
