#!/usr/bin/env python3
"""Stop the feet skating, without touching the pose.

    MUJOCO_GL=egl .venv/bin/python tools/stabilise_root.py in.csv out.csv

The joint angles are the performance and are left alone. What moves is the root:
six degrees of freedom that decide where the robot is and which way it leans, and
which the retarget filled in with a lot of motion nobody asked for. T6's right foot
travels 249.7 cm along the floor while it is supposed to be planted, and the root
wanders a 213.9 cm path to finish 9.6 cm from where it started.

Three passes, in this order, because each depends on the one before:

  1. LEAN.   Rotate the whole body so the soles lie flat, using the feet's own
     up-axes. Changes which foot is down, so it goes first.
  2. ANCHOR. Translate in XY so a foot that is down stays where it was. This is the
     one that removes the skating. Rigid translation moves the centre of mass and
     the feet together, so it cannot affect balance — only where the robot stands.
  3. ANKLE.  A bounded nudge (MAX_ANKLE) to flatten a sole the lean could not.
     Deliberately small: training is there to absorb the rest.

The anchor is re-taken whenever a foot lifts and re-plants, so a genuine step still
moves the robot; only sliding is removed. The correction is smoothed before it is
applied, since a per-frame fix would put a velocity step into the root.
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_columns import joint_cols  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
DOWN_H = 0.025            # metres above the clip floor still counts as planted
MAX_ANKLE = np.deg2rad(4.0)

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
FEET = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}
SIDES = ("left", "right")


def corners(side):
    pts = []
    for g in FEET[side]:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def put(root, quat, q):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root, quat, q
    mujoco.mj_forward(model, data)


def level(root, quat, q, iters=5):
    """Rotate the whole robot until the soles are flat, then drop it onto the floor."""
    root = np.asarray(root, float).copy()
    for _ in range(iters):
        put(root, quat, q)
        ns, cs = [], []
        for s in SIDES:
            P = corners(s)
            cs.append(P[np.argsort(P[:, 2])[:4]].mean(axis=0))
            b = model.geom_bodyid[FEET[s][0]]
            ns.append(data.xmat[b].reshape(3, 3) @ np.array([0.0, 0.0, 1.0]))
        n = np.mean(ns, axis=0)
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        axis = np.cross(n, [0.0, 0.0, 1.0])
        sa = np.linalg.norm(axis)
        if sa < 1e-9:
            break
        ang = np.arctan2(sa, float(np.dot(n, [0.0, 0.0, 1.0])))
        if abs(ang) < 1e-5 or abs(ang) > np.deg2rad(45.0):
            break
        R = Rotation.from_rotvec(axis / sa * ang)
        c = np.mean(cs, axis=0)
        root = R.apply(root - c) + c
        quat = (R * Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    put(root, quat, q)
    root[2] -= min(corners(s)[:, 2].min() for s in SIDES)
    return root, quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    jc = joint_cols(df)
    idx = {c[:-4]: i for i, c in enumerate(jc)}
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]
    Q = np.deg2rad(df[jc].values)
    ROOT = df[tc].values / 100.0
    QUAT = Rotation.from_euler("xyz", df[rc].values, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    T = len(df)

    # ---- 1. lean -------------------------------------------------------------
    root_l = np.zeros_like(ROOT)
    quat_l = np.zeros_like(QUAT)
    for i in range(T):
        root_l[i], quat_l[i] = level(ROOT[i], QUAT[i], Q[i])

    # ---- 2. anchor -----------------------------------------------------------
    cen = np.zeros((T, 2, 2))
    low = np.zeros((T, 2))
    for i in range(T):
        put(root_l[i], quat_l[i], Q[i])
        for k, s in enumerate(SIDES):
            P = corners(s)
            cen[i, k] = P.mean(axis=0)[:2]
            low[i, k] = P[:, 2].min()
    floor = np.percentile(low.min(axis=1), 2)
    down = (low - floor) < DOWN_H

    shift = np.zeros((T, 2))
    anchor = {0: None, 1: None}
    run = np.zeros(2)
    for i in range(T):
        # When the pose itself moves the feet apart no rigid translation can hold
        # both, so this minimises TOTAL slide rather than pinning one foot. Two
        # alternatives were measured and are worse: anchoring only the lowest foot
        # gives 262 cm of total slide, weighting by contact firmness 242, plain
        # averaging 214.
        for k in range(2):
            if not down[i, k]:
                anchor[k] = None                          # lifted: forget it
        want, wts = [], []
        for k in range(2):
            if down[i, k]:
                if anchor[k] is None:
                    anchor[k] = cen[i, k] + run           # newly planted: take it here
                want.append(anchor[k] - (cen[i, k] + run))
        if want:
            run = run + np.mean(np.array(want), axis=0)
        shift[i] = run

    # smooth the correction: applied raw it is a velocity step in the root
    w = max(3, int(round(0.08 * args.fps)) | 1)   # short: a long window undoes the fix
    pad = np.pad(shift, ((w // 2, w // 2), (0, 0)), mode="edge")
    shift = np.stack([np.convolve(pad[:, j], np.ones(w) / w, mode="valid") for j in range(2)], 1)
    root_a = root_l.copy()
    root_a[:, :2] += shift

    # ---- 3. ankle ------------------------------------------------------------
    Q2 = Q.copy()
    for i in range(T):
        for s in SIDES:
            if not down[i, SIDES.index(s)]:
                continue
            for part in ("ankle_roll", "ankle_pitch"):
                col = idx[f"{s}_{part}_joint"]
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{part}_joint")
                jlo, jhi = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)
                base, best = Q2[i, col], None
                for d in np.linspace(-MAX_ANKLE, MAX_ANKLE, 17):
                    Q2[i, col] = float(np.clip(base + d, jlo, jhi))
                    put(root_a[i], quat_l[i], Q2[i])
                    P = corners(s)
                    s4 = np.sort(P[:, 2])[:4]
                    gap = s4.max() - s4.min()
                    if best is None or gap < best[0]:
                        best = (gap, Q2[i, col])
                Q2[i, col] = best[1]
    for i in range(T):
        put(root_a[i], quat_l[i], Q2[i])
        root_a[i, 2] -= min(corners(s)[:, 2].min() for s in SIDES)

    out = df.copy()
    out[jc] = np.rad2deg(Q2)
    out[tc] = root_a * 100.0
    out[rc] = Rotation.from_quat(quat_l[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")

    def slide_of(root, quat, q):
        pos = np.zeros((T, 2, 2))
        for i in range(T):
            put(root[i], quat[i], q[i])
            for k, s in enumerate(SIDES):
                pos[i, k] = corners(s).mean(axis=0)[:2]
        d = np.linalg.norm(np.diff(pos, axis=0), axis=2) * 100
        return np.where(down[1:], d, 0.0).sum(axis=0)

    b = slide_of(ROOT, QUAT, Q)
    a = slide_of(root_a, quat_l, Q2)
    print(f"{os.path.basename(args.in_csv)} -> {os.path.basename(args.out_csv)}")
    print(f"  foot slide while planted   L {b[0]:6.1f} -> {a[0]:5.1f} cm    "
          f"R {b[1]:6.1f} -> {a[1]:5.1f} cm")
    print(f"  root path                  {np.linalg.norm(np.diff(ROOT[:, :2], axis=0), axis=1).sum() * 100:6.1f}"
          f" -> {np.linalg.norm(np.diff(root_a[:, :2], axis=0), axis=1).sum() * 100:5.1f} cm")
    print(f"  ankle nudged by            max {np.rad2deg(np.abs(Q2 - Q).max()):.1f} deg "
          f"(bound {np.rad2deg(MAX_ANKLE):.0f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
