#!/usr/bin/env python3
"""Put the spear-throwing arm back on the horizontal.

    MUJOCO_GL=egl .venv/bin/python tools/level_throw_arm.py in.csv out.csv

The arm was levelled once already, back when the take was a human BVH
(tools/lower_throw_arm.py, the A2 edit): at full extension the throwing hand should
sit level with the shoulder, not point up. A9 still has it, at +1.2 degrees.

Re-seating the robot on the floor undoes it. Levelling rotates the WHOLE body to
put the soles flat, and the arm rotates with it — by the time the coil rebuild is
done the hand is +27.7 degrees, pointing at the sky. The pose is right in the
robot's own frame and wrong in the world's, which is the frame the throw reads in.

So the correction is re-applied here, in G1 joint space this time, with the same
three rules the BVH version needed:

  * weighted by how extended the arm is, so it fades in and out instead of
    stepping,
  * only ever LOWERING — an already-level frame is left alone,
  * confined to the extension run containing the peak, so the rest of the phrase,
    including the uppercut, is untouched.

Shoulder pitch is solved per frame by bisection against the elevation MuJoCo
reports, clamped to the joint's range, then smoothed.
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
EXT_LO, EXT_HI = 0.86, 0.96      # the weighting window, as in the BVH version
TARGET_ELEV = 0.0

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)


def bid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)


def elevation(root, quat, q, side="right"):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root, quat, q
    mujoco.mj_forward(model, data)
    sh = data.xpos[bid(f"{side}_shoulder_roll_link")]
    hd = data.xpos[bid(f"{side}_wrist_yaw_link")]
    v = hd - sh
    r = np.linalg.norm(v)
    return np.degrees(np.arcsin(np.clip(v[2] / max(r, 1e-9), -1, 1))), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--side", default="right")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--window", default=None,
                    help="lo,hi frames — use the OTHER arm's throw window rather than "
                         "this arm's own extension peak. The pull-back arm does not "
                         "reach its longest during the throw, so its own peak lands "
                         "somewhere else in the phrase entirely.")
    ap.add_argument("--both-ways", action="store_true",
                    help="raise as well as lower; the pull-back arm can be under the "
                         "horizontal as easily as over it")
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

    elev = np.zeros(T)
    reach = np.zeros(T)
    for i in range(T):
        elev[i], reach[i] = elevation(ROOT[i], QUAT[i], Q[i], args.side)

    if args.window:
        lo, hi = (int(x) for x in args.window.split(","))
        peak = lo + int(np.argmax(reach[lo:hi + 1]))
        # taper in and out over 15 frames so the correction does not step
        w = np.zeros(T)
        ramp = max(1, int(0.25 * args.fps))
        for i in range(lo, hi + 1):
            w[i] = min(1.0, (i - lo + 1) / ramp, (hi - i + 1) / ramp)
    else:
        peak = int(np.argmax(reach))
        rmax = reach[peak]
        w = np.clip((reach / rmax - EXT_LO) / (EXT_HI - EXT_LO), 0.0, 1.0)
        # only the extension run containing the peak, so the uppercut is untouched
        lo = peak
        while lo > 0 and w[lo - 1] > 1e-3:
            lo -= 1
        hi = peak
        while hi < T - 1 and w[hi + 1] > 1e-3:
            hi += 1
        w[:lo] = 0.0
        w[hi + 1:] = 0.0
    rmax = reach[peak]

    col = idx[f"{args.side}_shoulder_pitch_joint"]
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{args.side}_shoulder_pitch_joint")
    jlo, jhi = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)

    probe = Q[peak].copy()
    probe[col] = float(np.clip(Q[peak, col] + np.deg2rad(2.0), jlo, jhi))
    sign = 1.0 if elevation(ROOT[peak], QUAT[peak], probe, args.side)[0] < elev[peak] else -1.0

    delta = np.zeros(T)
    for i in range(lo, hi + 1):
        if w[i] < 1e-3:
            continue
        if not args.both_ways and elev[i] <= TARGET_ELEV:   # only ever lower
            continue
        a, b = (-np.deg2rad(45.0), np.deg2rad(45.0)) if args.both_ways else (0.0, np.deg2rad(45.0))
        fa = None
        for _ in range(24):
            mid = 0.5 * (a + b)
            q = Q[i].copy()
            q[col] = float(np.clip(Q[i, col] + sign * mid, jlo, jhi))
            e = elevation(ROOT[i], QUAT[i], q, args.side)[0]
            if fa is None:
                qa = Q[i].copy()
                qa[col] = float(np.clip(Q[i, col] + sign * a, jlo, jhi))
                fa = elevation(ROOT[i], QUAT[i], qa, args.side)[0] - TARGET_ELEV
            if (e - TARGET_ELEV) * fa > 0:
                a, fa = mid, e - TARGET_ELEV
            else:
                b = mid
        delta[i] = sign * 0.5 * (a + b) * w[i]

    k = max(3, int(round(0.12 * args.fps)) | 1)
    pad = np.pad(delta, (k // 2, k // 2), mode="edge")
    delta = np.convolve(pad, np.ones(k) / k, mode="valid")

    Q2 = Q.copy()
    Q2[:, col] = np.clip(Q[:, col] + delta, jlo, jhi)

    after = np.array([elevation(ROOT[i], QUAT[i], Q2[i], args.side)[0] for i in range(T)])
    out = df.copy()
    out[jc] = np.rad2deg(Q2)
    out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"{os.path.basename(args.in_csv)} -> {os.path.basename(args.out_csv)}")
    print(f"  extension run frames {lo}..{hi}, peak {peak} (reach {rmax * 100:.1f} cm)")
    print(f"  hand elevation at peak   {elev[peak]:+6.1f} -> {after[peak]:+5.1f} deg")
    print(f"  shoulder pitch moved by  max {np.rad2deg(np.abs(delta)).max():.1f} deg")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
