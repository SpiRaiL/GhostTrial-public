#!/usr/bin/env python3
"""Lean each pose until the robot could stand in it.

    MUJOCO_GL=egl .venv/bin/python tools/balance_pose.py in.csv out.csv --target 3

The reference asks for poses the G1 cannot hold: on take A4 only 29% of frames put
the centre of mass inside the polygon the feet make on the floor. A policy given
those poses has no option but to step, so it never reproduces the move.

The fix is the one a person uses without thinking: lean at the ankle. Rotating the
whole body about the foot contact centre moves the centre of mass horizontally by
roughly (CoM height) x (lean angle), and subtracting the same angle from both
ankles keeps the soles flat on the ground. Nothing above the ankles changes, so the
pose keeps its shape — the arms, the torso twist, the crouch all survive; the robot
just stands under itself instead of beside itself.

Iterated a few times per frame because the CoM height and the support polygon both
move as it leans. Where the ankle range runs out before the pose is balanced, that
is reported rather than silently clamped — it means the pose is not reachable by
leaning alone.
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
CONTACT_H = 0.02

ap = argparse.ArgumentParser()
ap.add_argument("in_csv")
ap.add_argument("out_csv")
ap.add_argument("--target", type=float, default=3.0, help="wanted margin, cm")
ap.add_argument("--iters", type=int, default=6)
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
floor_g = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
foot_g = [g for g in range(model.ngeom)
          if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                model.geom_bodyid[g]) or "")]
lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]

df = pd.read_csv(args.in_csv)
jc = [c for c in df.columns if c.endswith("_dof")]
jname = [c[:-4] for c in jc]
AP = [jname.index("left_ankle_pitch_joint"), jname.index("right_ankle_pitch_joint")]
AR = [jname.index("left_ankle_roll_joint"), jname.index("right_ankle_roll_joint")]

root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
eul = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values
dof = np.deg2rad(df[jc].values)
T = len(df)


def support_and_com(rp, rq, q):
    data.qpos[:3] = rp
    data.qpos[3:7] = rq
    data.qpos[7:] = q
    mujoco.mj_forward(model, data)
    pts = []
    for g in foot_g:
        pos, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for szz in (-1, 1):
                    pts.append(pos + R @ (np.array([sx, sy, szz]) * sz))
    pts = np.array(pts)
    ground = pts[:, 2].min()
    on = pts[pts[:, 2] < ground + CONTACT_H]
    return on, np.array(data.subtree_com[0]), ground


def margin_of(on, com):
    if len(on) < 3:
        return -np.inf, None
    try:
        hull = ConvexHull(on[:, :2])
    except Exception:
        return -np.inf, None
    inside, best = True, np.inf
    for eq in hull.equations:
        d = float(eq[:2] @ com[:2] + eq[2])
        if d > 0:
            inside = False
        best = min(best, abs(d))
    centre = on[hull.vertices, :2].mean(axis=0)
    return (best if inside else -best), centre


out = df.copy()
before, after, stuck = np.zeros(T), np.zeros(T), 0
for i in range(T):
    rp, e, q = root[i].copy(), eul[i].copy(), dof[i].copy()
    rq = Rotation.from_euler("xyz", e, degrees=True).as_quat()[[3, 0, 1, 2]]
    on, com, ground = support_and_com(rp, rq, q)
    m0, _ = margin_of(on, com)
    before[i] = m0

    # FIRST put the soles flat. Leaning cannot help a frame that has no contact
    # patch, and most frames have none: the retargeted feet are tilted, so
    # grounding balances the robot on a corner rather than a sole. Drive each
    # ankle so its foot's own up-axis is vertical, within the joint's range.
    # Numerical, not analytic: drive each ankle uphill on "how vertical is this
    # foot's up-axis". Two DOF, finite differences, a handful of steps. Working out
    # the sign conventions by hand got it wrong twice, and this cannot get them
    # wrong because it only ever moves in the direction that measurably helps.
    def foot_up(side):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        return float(data.xmat[b].reshape(3, 3)[2, 2])

    for _ in range(8):
        for side, kp, kr in (("left", AP[0], AR[0]), ("right", AP[1], AR[1])):
            for k in (kp, kr):
                data.qpos[:3] = rp; data.qpos[3:7] = rq; data.qpos[7:] = q
                mujoco.mj_forward(model, data); base = foot_up(side)
                eps = 0.02
                q[k] = float(np.clip(q[k] + eps, lo[k], hi[k]))
                data.qpos[7:] = q; mujoco.mj_forward(model, data)
                up_plus = foot_up(side)
                q[k] = float(np.clip(q[k] - eps, lo[k], hi[k]))
                grad = (up_plus - base) / eps
                q[k] = float(np.clip(q[k] + 0.25 * grad, lo[k], hi[k]))

    for _ in range(args.iters):
        on, com, ground = support_and_com(rp, rq, q)
        m, centre = margin_of(on, com)
        if centre is None or m >= args.target / 100.0:
            break
        pivot = np.array([on[:, 0].mean(), on[:, 1].mean(), ground])
        err = centre - com[:2]                       # move the CoM toward the middle
        h = max(com[2] - ground, 0.25)
        # lean angles: about +Y moves the CoM in +X, about -X moves it in +Y
        d_pitch = np.arctan2(err[0], h)
        d_roll = np.arctan2(err[1], h)
        step = 0.6
        rot = Rotation.from_rotvec([-d_roll * step, d_pitch * step, 0.0])
        # rotate the whole body about the foot centre
        rp = pivot + rot.apply(rp - pivot)
        rq_r = (rot * Rotation.from_quat(rq[[1, 2, 3, 0]])).as_quat()
        rq = rq_r[[3, 0, 1, 2]]
        # ...and take the same angle out of the ankles so the soles stay flat
        for k in AP:
            q[k] = np.clip(q[k] - d_pitch * step, lo[k], hi[k])
        for k in AR:
            q[k] = np.clip(q[k] + d_roll * step, lo[k], hi[k])

    # Re-seat on the floor. Leaning rotates the body about the foot centre, which
    # lifts the feet clear of the ground unless the root is dropped again — the
    # contact-schedule check showed 61% of the "balanced" output was actually in
    # flight, which makes the balance figure meaningless.
    data.qpos[:3] = rp; data.qpos[3:7] = rq; data.qpos[7:] = q
    mujoco.mj_forward(model, data)
    lowest = min(float(p_[2]) for g in foot_g
                 for p_ in [data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3)
                            @ (np.array([sx, sy, sz_]) * model.geom_size[g])
                            for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)])
    rp[2] -= lowest

    on, com, ground = support_and_com(rp, rq, q)
    m1, _ = margin_of(on, com)
    after[i] = m1
    if m1 < args.target / 100.0:
        stuck += 1

    out.loc[i, ["root_translateX", "root_translateY", "root_translateZ"]] = rp * 100.0
    out.loc[i, ["root_rotateX", "root_rotateY", "root_rotateZ"]] = \
        Rotation.from_quat(rq[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    out.loc[i, jc] = np.degrees(q)

fin_b = before[np.isfinite(before)]
fin_a = after[np.isfinite(after)]
print(f"{os.path.basename(args.in_csv)}  {T} frames")
print(f"  balance margin  median {np.median(fin_b) * 100:+6.1f} -> {np.median(fin_a) * 100:+6.1f} cm")
print(f"  holdable frames {100 * (before > 0).mean():5.0f}% -> {100 * (after > 0).mean():5.0f}%")
print(f"  still short of +{args.target:g} cm: {stuck} frames ({100 * stuck / T:.0f}%)")
out.to_csv(args.out_csv, index=False, float_format="%.6f")
print(f"wrote {args.out_csv}")
