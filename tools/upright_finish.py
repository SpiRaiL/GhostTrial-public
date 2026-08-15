#!/usr/bin/env python3
"""Stand the robot up over its feet after the uppercut, so it cannot fall backwards.

    MUJOCO_GL=egl .venv/bin/python tools/upright_finish.py in.csv out.csv

At the top of the uppercut the body is leaning back: the centre of mass sits 13.9 cm
BEHIND the centre of the feet and the torso is 24 degrees off vertical, and it stays
behind the feet for 87% of what follows. A tracking policy asked to hold that will
either fall backwards or refuse the swing — which is the likely reason the uppercut
went missing from the trained rollouts.

Nothing about the pose changes. The whole robot is pitched about the contact
centroid, which moves the centre of mass forward over the feet while leaving the feet
where they are, and then dropped back onto the floor. Pitching forward reduces the
backward lean and the CoM offset together — they are the same error seen two ways.

The correction ramps in and out around the uppercut so the rest of the phrase is
untouched, is bounded by MAX_PITCH, and is smoothed before it is applied.
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
MAX_PITCH = np.deg2rad(20.0)
TARGET_OFFSET = 0.0            # metres: centre of mass over the middle of the feet

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


def state(root, quat, q):
    """Fore/aft offset of the centre of mass from the foot centre, and torso tilt."""
    put(root, quat, q)
    P = np.vstack([corners(s) for s in SIDES])
    com = data.subtree_com[0]
    yaw = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("zyx")[0]
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    off = float((com[:2] - P[:, :2].mean(axis=0)) @ fwd)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    tz = data.xmat[b].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    tilt = np.degrees(np.arccos(np.clip(tz[2], -1, 1)))
    return off, tilt, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--from-frame", type=int, default=None,
                    help="default: the uppercut peak, found as the highest hand")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    jc = joint_cols(df)
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]
    Q = np.deg2rad(df[jc].values)
    ROOT = df[tc].values / 100.0
    QUAT = Rotation.from_euler("xyz", df[rc].values, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    T = len(df)

    if args.from_frame is None:
        hand = np.zeros(T)
        for i in range(T):
            put(ROOT[i], QUAT[i], Q[i])
            hand[i] = max(data.xpos[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link")][2] for s in SIDES)
        start = int(np.argmax(hand[T // 2:])) + T // 2
    else:
        start = args.from_frame

    ramp = max(1, int(0.5 * args.fps))
    weight = np.zeros(T)
    for i in range(T):
        if i >= start - ramp:
            weight[i] = min(1.0, (i - (start - ramp)) / ramp)

    off0 = np.array([state(ROOT[i], QUAT[i], Q[i])[0] for i in range(T)])

    delta = np.zeros(T)
    for i in range(T):
        if weight[i] < 1e-3 or off0[i] >= TARGET_OFFSET:
            continue                                   # already over or ahead of the feet
        lo, hi = 0.0, MAX_PITCH
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            put(ROOT[i], QUAT[i], Q[i])
            P = np.vstack([corners(s) for s in SIDES])
            c = P[np.argsort(P[:, 2])[:8]].mean(axis=0)
            yaw = Rotation.from_quat(QUAT[i][[1, 2, 3, 0]]).as_euler("zyx")[0]
            axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])   # pitch about the feet
            R = Rotation.from_rotvec(axis * mid)
            r2 = R.apply(ROOT[i] - c) + c
            q2 = (R * Rotation.from_quat(QUAT[i][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
            if state(r2, q2, Q[i])[0] < TARGET_OFFSET:
                lo = mid
            else:
                hi = mid
        delta[i] = hi * weight[i]

    k = max(3, int(round(0.25 * args.fps)) | 1)
    pad = np.pad(delta, (k // 2, k // 2), mode="edge")
    delta = np.convolve(pad, np.ones(k) / k, mode="valid")

    root2 = ROOT.copy()
    quat2 = QUAT.copy()
    for i in range(T):
        if delta[i] < 1e-6:
            continue
        put(ROOT[i], QUAT[i], Q[i])
        P = np.vstack([corners(s) for s in SIDES])
        c = P[np.argsort(P[:, 2])[:8]].mean(axis=0)
        yaw = Rotation.from_quat(QUAT[i][[1, 2, 3, 0]]).as_euler("zyx")[0]
        R = Rotation.from_rotvec(np.array([-np.sin(yaw), np.cos(yaw), 0.0]) * delta[i])
        root2[i] = R.apply(ROOT[i] - c) + c
        quat2[i] = (R * Rotation.from_quat(QUAT[i][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
        put(root2[i], quat2[i], Q[i])
        root2[i, 2] -= min(corners(s)[:, 2].min() for s in SIDES)

    o1 = np.array([state(root2[i], quat2[i], Q[i])[0] for i in range(T)])
    t0 = np.array([state(ROOT[i], QUAT[i], Q[i])[1] for i in range(T)])
    t1 = np.array([state(root2[i], quat2[i], Q[i])[1] for i in range(T)])

    out = df.copy()
    out[tc] = root2 * 100.0
    out[rc] = Rotation.from_quat(quat2[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    sl = slice(start, T)
    print(f"{os.path.basename(args.in_csv)} -> {os.path.basename(args.out_csv)}")
    print(f"  uppercut peak frame {start} ({start / 30:.1f}s of video)")
    print(f"  CoM behind the feet   {100 * (off0[sl] < 0).mean():3.0f}% -> "
          f"{100 * (o1[sl] < 0).mean():3.0f}% of frames after it")
    print(f"  worst offset          {off0[sl].min() * 100:+6.1f} -> {o1[sl].min() * 100:+5.1f} cm")
    print(f"  torso tilt at peak    {t0[start]:6.1f} -> {t1[start]:5.1f} deg from vertical")
    print(f"  body pitched by       max {np.rad2deg(delta).max():.1f} deg (bound "
          f"{np.rad2deg(MAX_PITCH):.0f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
