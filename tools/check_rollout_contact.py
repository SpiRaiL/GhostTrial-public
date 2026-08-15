#!/usr/bin/env python3
"""Did the policy actually put its feet on the floor? Rollout vs its own target.

    MUJOCO_GL=egl .venv/bin/python tools/check_rollout_contact.py \
        data/rollouts/s1_policy.csv data/motion_lib_capture/robot/s1/S1_stand.pkl

The gap this closes: the frozen-stance run reported 3.9 deg mean joint error and
99.4% of episodes completing, and the policy was standing on ONE foot with the
other hovering 6.3 cm off the floor for every frame of the clip. Neither number
can see that. A hovering foot is a few degrees of ankle and hip, so it barely
moves the joint error; and "episode completed" only means the robot never fell.

So contact is measured directly, per foot, and against the target's own contact
schedule rather than an absolute rule — if the reference stands on one leg, the
policy should too.

Reported per foot:
  clearance   sole height above the floor the clip itself establishes
  down        fraction of frames that foot is loaded (within 2 cm)
  agreement   fraction of frames the policy's contact matches the target's

Also stance width, because a policy that shows correct contact while splaying
its feet has not tracked the pose either.
"""

import argparse
import os

import joblib
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from g1_columns import joint_cols, was_reordered

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
DOWN = 0.02               # metres above the clip's floor still counts as loaded

ap = argparse.ArgumentParser()
ap.add_argument("rollout_csv")
ap.add_argument("target", help="motion_lib .pkl the policy was tracking")
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
feet = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}


def corners(geoms):
    pts = []
    for g in geoms:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def trace(root, quat, dof):
    """Sole height and stance width per frame."""
    T = len(dof)
    low = np.zeros((T, 2))
    width = np.zeros(T)
    for i in range(T):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], dof[i]
        mujoco.mj_forward(model, data)
        L, R = corners(feet["left"]), corners(feet["right"])
        low[i] = L[:, 2].min(), R[:, 2].min()
        yaw = Rotation.from_quat(quat[i][[1, 2, 3, 0]]).as_euler("zyx")[0]
        Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        width[i] = abs((Rz @ L.mean(0)[:2])[1] - (Rz @ R.mean(0)[:2])[1])
    # each clip establishes its own floor: the lowest any sole ever gets
    return low - low.min(), width


tgt = joblib.load(args.target)
t = tgt[list(tgt)[0]]
t_low, t_width = trace(np.asarray(t["root_trans_offset"]),
                       np.asarray(t["root_rot"])[:, [3, 0, 1, 2]],
                       np.asarray(t["dof"]))

df = pd.read_csv(args.rollout_csv)
jc = joint_cols(df)
p_low, p_width = trace(
    df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0,
    Rotation.from_euler("xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
                        degrees=True).as_quat()[:, [3, 0, 1, 2]],
    np.deg2rad(df[jc].values))

# Resample onto a COMMON TIMELINE. The target is a motion_lib clip at its own fps,
# the rollout is one row per control step at 50 Hz — comparing index to index lines
# up frame 100 of a 30 fps clip with 2.0 s of a 50 Hz one against 3.3 s, and every
# contact number computed that way was comparing different moments of the move.
t_fps = float(tgt[list(tgt)[0]].get("fps", 30))
p_fps = float(os.environ.get("GT_POLICY_FPS", 50))
dur = min(len(t_low) / t_fps, len(p_low) / p_fps)
n = max(2, int(dur * t_fps))
ts = np.linspace(0, dur, n)
ti = np.clip((ts * t_fps).astype(int), 0, len(t_low) - 1)
pi = np.clip((ts * p_fps).astype(int), 0, len(p_low) - 1)
t_low, p_low = t_low[ti], p_low[pi]
t_width, p_width = t_width[ti], p_width[pi]
t_down, p_down = t_low < DOWN, p_low < DOWN

print(f"\n\033[1m{os.path.basename(args.rollout_csv)}\033[0m vs "
      f"{os.path.basename(args.target)}   ({dur:.2f} s compared, "
      f"target {t_fps:g} fps vs policy {p_fps:g} Hz)")
print(f"{'':14s}{'target':>22s}{'policy':>22s}")
for s, k in (("left", 0), ("right", 1)):
    print(f"  {s + ' clearance':13s}"
          f"{t_low[:, k].mean() * 100:8.1f} cm mean{'':<5}"
          f"{p_low[:, k].mean() * 100:8.1f} cm mean")
    ok = 100 * p_down[:, k].mean()
    flag = "\033[32m" if ok > 90 else "\033[31m"
    print(f"  {s + ' down':13s}{100 * t_down[:, k].mean():8.0f}% of frames{'':<3}"
          f"{flag}{ok:8.0f}% of frames\033[0m")

agree = 100 * (t_down == p_down).all(axis=1).mean()
both_t, both_p = 100 * t_down.all(axis=1).mean(), 100 * p_down.all(axis=1).mean()
print(f"  {'double support':13s}{both_t:8.0f}%{'':<12}{both_p:8.0f}%")
print(f"  {'stance width':13s}{t_width.mean() * 100:8.1f} cm{'':<10}"
      f"{p_width.mean() * 100:8.1f} cm")
print(f"\n  \033[1mcontact schedule matches the target on {agree:.0f}% of frames\033[0m")
if both_p < 90 <= both_t:
    print("  \033[31m^ the reference stands on two feet and the policy does not — "
          "it is balancing, not standing\033[0m")
