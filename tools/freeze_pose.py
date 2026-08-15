#!/usr/bin/env python3
"""Freeze one frame of a G1 motion into a still clip, and report the stance it asks for.

    MUJOCO_GL=egl .venv/bin/python tools/freeze_pose.py in.csv out.csv [--seconds 4]

Why this exists: the frozen opening stance (S1) reported 99.4% of episodes
completing and 3.9 deg mean joint error, and still the policy stood on ONE foot
with the other hovering 6.3 cm above the floor for the whole clip. Aggregate joint
error cannot see that — a hovering foot is only a few degrees of ankle and hip.
So the stance geometry is measured here, at the point the clip is made, and printed
in the character's own frame rather than the world's:

    stride         fore/aft separation of the feet
    lateral width  side-to-side separation      <- the one that mattered
    CoM offset     how far the centre of mass sits off the midline between the feet

Our opening asks for a 57 cm straddle. The motions SONIC already tracks sit at
32-36 cm. That is the defect: a stance 60% wider than anything in the training
distribution, held statically, needs continuous hip abduction torque, and the
policy sheds it by unweighting a foot.

--pick still chooses the frame for you: among frames with both feet down, the one
with the lowest joint velocity and the most centred centre of mass.
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
CONTACT_H = 0.02          # metres above the clip's own floor to count as down

ap = argparse.ArgumentParser()
ap.add_argument("in_csv")
ap.add_argument("out_csv")
ap.add_argument("--frame", type=int, default=-1, help="frame to freeze; -1 = pick")
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--fps", type=float, default=120.0, help="BONES-SEED rate")
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
root = df[tc].values / 100.0
quat = Rotation.from_euler("xyz", df[rc].values, degrees=True).as_quat()[:, [3, 0, 1, 2]]
dof = np.deg2rad(df[jc].values)
T = len(df)


def sole_corners(geoms):
    pts = []
    for g in geoms:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def measure(i):
    """Stance geometry at frame i, in the pelvis's own yaw frame."""
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], dof[i]
    mujoco.mj_forward(model, data)
    L, R = sole_corners(feet["left"]), sole_corners(feet["right"])
    yaw = Rotation.from_quat(quat[i][[1, 2, 3, 0]]).as_euler("zyx")[0]
    Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    lc, rc_, com = Rz @ L.mean(0)[:2], Rz @ R.mean(0)[:2], Rz @ data.subtree_com[0][:2]
    return dict(stride=abs(lc[0] - rc_[0]), width=abs(lc[1] - rc_[1]),
                com_off=com[1] - (lc[1] + rc_[1]) / 2,
                low_l=L[:, 2].min(), low_r=R[:, 2].min())


s = [measure(i) for i in range(T)]
floor = min(min(x["low_l"] for x in s), min(x["low_r"] for x in s))
both = np.array([(x["low_l"] - floor < CONTACT_H) and (x["low_r"] - floor < CONTACT_H)
                 for x in s])

if args.frame >= 0:
    pick = args.frame
else:
    vel = np.zeros(T)
    vel[1:-1] = np.abs(dof[2:] - dof[:-2]).sum(axis=1)
    vel[0], vel[-1] = vel[1], vel[-2]
    off = np.array([abs(x["com_off"]) for x in s])
    score = np.where(both, vel / (vel.max() + 1e-9) + off / (off.max() + 1e-9), np.inf)
    pick = int(np.argmin(score))
    if not np.isfinite(score[pick]):
        raise SystemExit("no frame has both feet down")

g = s[pick]
n = int(round(args.seconds * args.fps))
out = pd.DataFrame(np.repeat(df.iloc[[pick]].values, n, axis=0), columns=df.columns)
# re-seat so the lower sole rests on z=0 rather than wherever the clip left it
out[tc[2]] -= min(g["low_l"], g["low_r"]) * 100.0
out.to_csv(args.out_csv, index=False, float_format="%.6f")

print(f"{os.path.basename(args.in_csv)}  {T} frames, {int(both.sum())} with both feet down")
print(f"  froze frame {pick} -> {n} frames ({args.seconds:g} s at {args.fps:g} fps)")
print(f"  stride         {g['stride'] * 100:5.1f} cm")
print(f"  lateral width  {g['width'] * 100:5.1f} cm    "
      f"(motions SONIC tracks: 32-36; our opening: 57)")
print(f"  CoM off midline{g['com_off'] * 100:+5.1f} cm")
print(f"wrote {args.out_csv}")
