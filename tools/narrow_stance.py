#!/usr/bin/env python3
"""Bring an over-wide stance back into the training distribution.

    MUJOCO_GL=egl .venv/bin/python tools/narrow_stance.py in.csv out.csv --width 30

Our opening asks for a 57 cm lateral straddle. Every motion SONIC actually tracks
sits at 32-36 cm. Held statically that needs continuous hip abduction with almost
no lateral margin, and the trained policy sheds the cost by unweighting a foot —
it stood on one leg for 100% of the frozen-stance rollout.

The knob is hip roll, applied symmetrically. It is solved per frame by bisection
against the width actually measured in MuJoCo rather than computed from a formula,
because hip roll, knee and ankle interact and a closed form would be a second
kinematics implementation to disagree with the one everything else is checked
against.

Two things learned the hard way and enforced here:

  * an earlier BVH-space attempt (stabilise_stance.py --narrow 3) saturated its
    own +-0.9 rad clamp and flung the legs to 78 degrees from vertical. The
    correction is bounded to MAX_DELTA and the bound being hit is reported, not
    silently accepted.
  * changing leg joints moves the FEET, not the pelvis, so the clip floats. The
    result is re-seated at the end, and the check is on the soles.
"""

import argparse
import os

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
MAX_DELTA = np.deg2rad(18.0)      # per hip; beyond this the pose stops being the pose

ap = argparse.ArgumentParser()
ap.add_argument("in_csv")
ap.add_argument("out_csv")
ap.add_argument("--width", type=float, default=32.0, help="target lateral width, cm")
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
feet = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}

df = pd.read_csv(args.in_csv)
jc = [c for c in df.columns if c.endswith("_dof")]
tc = ["root_translateX", "root_translateY", "root_translateZ"]
rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]
iL, iR = jc.index("left_hip_roll_joint_dof"), jc.index("right_hip_roll_joint_dof")

root = df[tc].values / 100.0
quat = Rotation.from_euler("xyz", df[rc].values, degrees=True).as_quat()[:, [3, 0, 1, 2]]
dof = np.deg2rad(df[jc].values)
T = len(df)


def corners(geoms):
    pts = []
    for g in geoms:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def state(i, delta):
    """Lateral width and lowest sole, with +-delta added to the hip rolls."""
    q = dof[i].copy()
    q[iL] -= delta
    q[iR] += delta
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], q
    mujoco.mj_forward(model, data)
    L, R = corners(feet["left"]), corners(feet["right"])
    yaw = Rotation.from_quat(quat[i][[1, 2, 3, 0]]).as_euler("zyx")[0]
    Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    w = abs((Rz @ L.mean(0)[:2])[1] - (Rz @ R.mean(0)[:2])[1])
    return w, min(L[:, 2].min(), R[:, 2].min()), q


target = args.width / 100.0
delta = np.zeros(T)
w0 = np.zeros(T)
w1 = np.zeros(T)
hit = 0
for i in range(T):
    w0[i], _, _ = state(i, 0.0)
    if w0[i] <= target:                       # already narrow enough; leave it alone
        w1[i] = w0[i]
        continue
    # sign check: which direction closes the stance?
    probe, _, _ = state(i, np.deg2rad(1.0))
    s = 1.0 if probe < w0[i] else -1.0
    lo, hi = 0.0, MAX_DELTA
    for _ in range(24):                       # bisection on measured width
        mid = 0.5 * (lo + hi)
        w, _, _ = state(i, s * mid)
        if w > target:
            lo = mid
        else:
            hi = mid
    delta[i] = s * hi
    w1[i], _, _ = state(i, delta[i])
    if hi >= MAX_DELTA - 1e-6:
        hit += 1

out = df.copy()
out[jc[iL]] = np.rad2deg(dof[:, iL] - delta)
out[jc[iR]] = np.rad2deg(dof[:, iR] + delta)

# re-seat: leg edits move the feet, so the clip floats unless the soles are put back
lows = np.array([state(i, delta[i])[1] for i in range(T)])
out[tc[2]] -= lows.min() * 100.0

out.to_csv(args.out_csv, index=False, float_format="%.6f")
print(f"{os.path.basename(args.in_csv)}  {T} frames, target width {args.width:g} cm")
print(f"  lateral width   {w0.mean() * 100:5.1f} -> {w1.mean() * 100:5.1f} cm "
      f"(max {w0.max() * 100:.1f} -> {w1.max() * 100:.1f})")
print(f"  hip roll change  mean {np.rad2deg(np.abs(delta)).mean():4.1f} deg per hip, "
      f"max {np.rad2deg(np.abs(delta)).max():4.1f}")
print(f"  re-seated by    {-lows.min() * 100:+5.1f} cm")
if hit:
    print(f"  \033[31m{hit}/{T} frames hit the {np.rad2deg(MAX_DELTA):.0f} deg bound "
          f"and are still too wide\033[0m")
print(f"wrote {args.out_csv}")
