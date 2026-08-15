#!/usr/bin/env python3
"""Author the Dex3 hand track that runs alongside the body policy.

    MUJOCO_GL=egl .venv/bin/python tools/author_hands.py body.csv out_hands.csv

SONIC's action space is 29 DoF and contains no fingers, so the hands cannot be
trained into the policy without changing that space and retraining from scratch.
They do not need to be: on a real G1 the Dex3 hands take direct position commands
and are driven separately from the whole-body controller. So this is a second track,
synchronised to the body motion, exactly as the robot would run it.

The rule, from David: fist throughout, open only for the spear throw and the pull
back. Both hands open together — the throw and the pull are one gesture.

The throw window is found from the body track rather than hard-coded, because the
take has been retimed twice (the pause cut to 1 s, then the whole clip to 1.5x) and
any frame number written down earlier is now wrong.

Dex3 is 7 joints per hand and the two hands are NOT mirror images in joint space:
the left index/middle ranges run -105..11 and -120..0, the right run -11..105 and
0..120. Signs are taken from each joint's own range rather than assumed, which is
the same trap that produced a backwards foot-flattening fix earlier in this project.
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
BODY_XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                        "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
HAND_XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic_deploy",
                        "g1", "g1_29dof_with_hand.xml")
OPEN_FRAC = 0.05        # how far toward the open end of each joint's range
FIST_FRAC = 1.00        # how far toward the closed end — a fist is fully curled

body = mujoco.MjModel.from_xml_path(BODY_XML)
bdata = mujoco.MjData(body)
hand = mujoco.MjModel.from_xml_path(HAND_XML)

HAND_JOINTS = [mujoco.mj_id2name(hand, mujoco.mjtObj.mjOBJ_JOINT, j)
               for j in range(1, hand.njnt)
               if "hand" in (mujoco.mj_id2name(hand, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]

# Which end of each joint's range curls the hand shut, MEASURED rather than assumed.
# Driving every joint of a hand to one end and comparing mean fingertip-to-palm
# distance answers it outright: left curls toward its minimum (11.4 cm against 15.5),
# right toward its maximum. They are mirrored, so a single table written by hand gets
# one side backwards — the first attempt curled the fingers but splayed both thumbs,
# and the fist read as a claw.
def _curl_signs():
    d = mujoco.MjData(hand)
    bods = [mujoco.mj_id2name(hand, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(hand.nbody)]
    signs = {}
    for side in ("left", "right"):
        js = [n for n in HAND_JOINTS if n.startswith(side)]
        dist = {}
        for end in (0, 1):
            d.qpos[:] = 0.0
            d.qpos[3] = 1.0
            for n in js:
                j = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_JOINT, n)
                d.qpos[hand.jnt_qposadr[j]] = hand.jnt_range[j][end]
            mujoco.mj_forward(hand, d)
            palm = d.xpos[mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_BODY,
                                            f"{side}_wrist_yaw_link")]
            tips = [b for b in bods if b.startswith(f"{side}_hand_") and b.endswith("_1_link")]
            dist[end] = float(np.mean([
                np.linalg.norm(d.xpos[mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_BODY, t)] - palm)
                for t in tips]))
        curl_end = 0 if dist[0] < dist[1] else 1
        for n in js:
            signs[n] = -1 if curl_end == 0 else +1
    return signs


CURL_SIGN = _curl_signs()


def pose(frac):
    """frac 0 = fully open, 1 = fully curled."""
    out = {}
    for n in HAND_JOINTS:
        j = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = hand.jnt_range[j]
        s = CURL_SIGN.get(n, 0)
        if s == 0:                       # thumb abduction: hold mid-range
            out[n] = 0.5 * (lo + hi)
        elif s > 0:
            out[n] = lo + frac * (hi - lo)
        else:
            out[n] = hi - frac * (hi - lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--open-lead", type=float, default=0.25,
                    help="seconds before/after the throw to blend the hands open")
    args = ap.parse_args()

    df = pd.read_csv(args.body_csv)
    jc = joint_cols(df)
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    T = len(df)

    # find the throw from the body track: the frame the throwing arm is longest
    reach = np.zeros(T)
    for i in range(T):
        bdata.qpos[:3], bdata.qpos[3:7], bdata.qpos[7:] = root[i], quat[i], dof[i]
        mujoco.mj_forward(body, bdata)
        sh = bdata.xpos[mujoco.mj_name2id(body, mujoco.mjtObj.mjOBJ_BODY,
                                          "right_shoulder_roll_link")]
        hd = bdata.xpos[mujoco.mj_name2id(body, mujoco.mjtObj.mjOBJ_BODY,
                                          "right_wrist_yaw_link")]
        reach[i] = np.linalg.norm(hd - sh)
    peak = int(np.argmax(reach))
    thr = reach.min() + 0.86 * (reach.max() - reach.min())
    lo = peak
    while lo > 0 and reach[lo - 1] > thr:
        lo -= 1
    hi = peak
    while hi < T - 1 and reach[hi + 1] > thr:
        hi += 1

    lead = max(1, int(args.open_lead * args.fps))
    openness = np.zeros(T)                      # 1 = open, 0 = fist
    for i in range(T):
        if lo <= i <= hi:
            openness[i] = 1.0
        elif lo - lead <= i < lo:
            openness[i] = (i - (lo - lead)) / lead
        elif hi < i <= hi + lead:
            openness[i] = 1.0 - (i - hi) / lead
    openness = np.clip(openness, 0, 1)

    fist, opened = pose(FIST_FRAC), pose(OPEN_FRAC)
    out = pd.DataFrame({"Frame": np.arange(T)})
    for n in HAND_JOINTS:
        out[f"{n}_dof"] = np.rad2deg(fist[n] * (1 - openness) + opened[n] * openness)
    out["hand_openness"] = openness
    out.to_csv(args.out_csv, index=False, float_format="%.6f")

    print(f"{os.path.basename(args.body_csv)} -> {os.path.basename(args.out_csv)}")
    print(f"  {len(HAND_JOINTS)} Dex3 joints, {T} frames")
    print(f"  throw window frames {lo}..{hi} (peak {peak}), "
          f"{(hi - lo + 1) / args.fps:.2f} s open")
    print(f"  hands open on {100 * (openness > 0.5).mean():.0f}% of the clip, "
          f"fist on {100 * (openness < 0.5).mean():.0f}%")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
