#!/usr/bin/env python3
"""Blend our crouch toward the BONES-SEED crouch, lower body only, and lay the series out.

    MUJOCO_GL=egl .venv/bin/python tools/blend_crouch.py

The performance we want lives in the arms and head; the posture the robot can hold
lives in a clip SONIC already tracks. So take each from where it is good:

    arms, head        A9 — the captured performance
    waist / back      REF_crouch — bent over, which is the look we want anyway
    hips, knees, ankles   blended REF_crouch -> A9 by t

t = 0 is the seed crouch outright: pelvis 62.4 cm, knee/hip 1.44, feet 35.7 cm
apart, centre of mass dead centre. t = 1 is our own crouch: pelvis 80.1 cm,
knee/hip 0.74, feet 52.6 cm apart, and a back bent at the WAIST rather than a
squat. Somewhere between is a pose that reads as the coil and can still be held.

The feet are allowed to move — they are an output of the blend, not a constraint.
Every blend is re-seated so the lower sole rests on the floor, because leg edits
move the feet and forgetting that has floated four takes in this project.

Angles are blended as angles, not as slerped rotations: these are independent
revolute joints well inside their ranges, where a linear blend is exact, and the
root orientation is carried through untouched.
"""

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
A9 = "data/gemx_g1_retimed/A9_smooth.csv"
REF = "data/csv_frozen/REF_crouch.csv"
A9_CROUCH_FRAME = 609          # deepest lean in A9, found by hip-pitch extremum

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
FEET = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}

LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST = ("waist_yaw", "waist_roll", "waist_pitch")


def corners(side):
    pts = []
    for g in FEET[side]:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def pose_stats(root, quat, dof):
    from scipy.spatial import ConvexHull
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root, quat, dof
    mujoco.mj_forward(model, data)
    L, R = corners("left"), corners("right")
    com = data.subtree_com[0]
    yaw = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("zyx")[0]
    Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    lc, rc = Rz @ L.mean(0)[:2], Rz @ R.mean(0)[:2]
    low = min(L[:, 2].min(), R[:, 2].min())
    down = [P for P in (L, R) if P[:, 2].min() - low < 0.02]
    margin = -np.inf
    if down:
        P = np.vstack([p[:, :2] for p in down])
        try:
            hull = ConvexHull(P)
            inside, best = True, np.inf
            for eq in hull.equations:
                dd = float(eq[:2] @ com[:2] + eq[2])
                if dd > 0:
                    inside = False
                best = min(best, abs(dd))
            margin = best if inside else -best
        except Exception:
            pass
    # how flat each sole lies: spread of its four lowest corners
    flat = max(np.ptp(np.sort(L[:, 2])[:4]), np.ptp(np.sort(R[:, 2])[:4]))
    return dict(pelvis=data.qpos[2] * 100, lateral=abs(lc[1] - rc[1]) * 100,
                stride=abs(lc[0] - rc[0]) * 100, margin=margin * 100, low=low,
                flat=flat * 100, com_off=(Rz @ com[:2])[1] * 100 - (lc[1] + rc[1]) / 2 * 100)


def seat_pose(root, quat, q, iters=6):
    """Place the robot so both soles lie on the floor, WITHOUT touching the pose.

    David: "we are not re-planting the feet back on the ground by means of rotating
    and translating the entire robot". Exactly right — the joint angles are the pose
    we are choosing, and the root is free. An earlier version opened the trailing
    leg's knee to reach the ground, which quietly rewrote the blend it was supposed
    to be evaluating.

    So the joints are held fixed and only the root 6-DoF moves: fit a plane through
    the sole contact points, rotate the whole body until that plane is horizontal,
    then drop it until it touches. Rotating about the contact centroid keeps the feet
    where they are instead of swinging the robot across the floor. A few iterations,
    because rotating changes which corners are lowest.
    """
    data.qpos[7:] = q
    for _ in range(iters):
        data.qpos[:3], data.qpos[3:7] = root, quat
        mujoco.mj_forward(model, data)
        # Average the two FEET'S OWN up-axes rather than fitting a plane to the
        # contact points. The point-cloud fit degenerates whenever those points are
        # nearly collinear — feet close together, or one sole dominating — and it
        # rotated the robot face-down at two blends (pelvis 61 cm -> 28 cm). A foot's
        # orientation is always well defined.
        ns, cs = [], []
        for side in ("left", "right"):
            P = corners(side)
            cs.append(P[np.argsort(P[:, 2])[:4]].mean(axis=0))
            b = model.geom_bodyid[FEET[side][0]]
            ns.append(data.xmat[b].reshape(3, 3) @ np.array([0.0, 0.0, 1.0]))
        n = np.mean(ns, axis=0)
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        c = np.mean(cs, axis=0)
        axis = np.cross(n, [0.0, 0.0, 1.0])
        s_ax = np.linalg.norm(axis)
        if s_ax < 1e-9:
            break
        angle = np.arctan2(s_ax, float(np.dot(n, [0.0, 0.0, 1.0])))
        if abs(angle) < 1e-5:
            break
        if abs(angle) > np.deg2rad(60.0):    # a sane pose never needs this much
            print(f"    [seat] refusing a {np.rad2deg(angle):.0f} deg correction")
            break
        R = Rotation.from_rotvec(axis / s_ax * angle)
        # rotate the whole robot about the contact centroid, so the feet stay put
        root = R.apply(root - c) + c
        quat = (R * Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    data.qpos[:3], data.qpos[3:7] = root, quat
    mujoco.mj_forward(model, data)
    root = root.copy()
    root[2] -= min(corners("left")[:, 2].min(), corners("right")[:, 2].min())
    return root, quat


def main():
    a9 = pd.read_csv(A9)
    ref = pd.read_csv(REF)
    jc = joint_cols(a9)
    idx = {c[:-4]: i for i, c in enumerate(jc)}
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]

    qa = np.deg2rad(a9[jc].values[A9_CROUCH_FRAME])
    qr = np.deg2rad(ref[joint_cols(ref)].values[0])
    root_a = a9[tc].values[A9_CROUCH_FRAME] / 100.0
    rot_a = Rotation.from_euler("xyz", a9[rc].values[A9_CROUCH_FRAME], degrees=True)
    rot_r = Rotation.from_euler("xyz", ref[rc].values[0], degrees=True)
    # The ROOT must blend with the legs. The seed crouch's leg angles are authored
    # under a pelvis pitched 37.5 deg forward; A9's crouch pelvis is at 13.7. Grafting
    # the joints alone tilts the whole body and lifts a foot — the first version of
    # this scored the seed crouch at -6.7 cm margin when on its own root it is +8.3.
    # Slerp, because root orientation is a rotation and Euler angles do not blend.
    # Blend the LEAN only, never the facing. REF_crouch faces -83.6 deg and A9 +90.4,
    # so slerping the full root orientation swung the character 174 degrees across the
    # series and every pose pointed somewhere different. Split each root into
    # yaw * tilt, keep A9's yaw throughout, and interpolate the tilt.
    def split(r):
        yaw = r.as_euler("zyx")[0]
        Rz = Rotation.from_euler("z", yaw)
        return Rz, Rz.inv() * r

    yaw_a, tilt_a = split(rot_a)
    _, tilt_r = split(rot_r)
    from scipy.spatial.transform import Slerp
    slerp_tilt = Slerp([0.0, 1.0], Rotation.concatenate([tilt_r, tilt_a]))

    leg_cols = [idx[f"{s}_{p}_joint"] for s in ("left", "right") for p in LEG]
    waist_cols = [idx[f"{w}_joint"] for w in WAIST]

    rows, poses = [], []
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        quat_a = (yaw_a * slerp_tilt([t])[0]).as_quat()[[3, 0, 1, 2]]
        q = qa.copy()                       # arms and head keep the performance
        for c in waist_cols:                # back bent like the seed crouch
            q[c] = qr[c]
        for c in leg_cols:                  # lower body blended
            q[c] = qr[c] * (1 - t) + qa[c] * t
        root, quat_seated = seat_pose(root_a.copy(), quat_a, q)
        s = pose_stats(root, quat_seated, q)
        # feasibility: is any joint outside its stop?
        over = 0
        for k, name in enumerate(jc):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name[:-4])
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                over += (q[k] < lo - 1e-6) or (q[k] > hi + 1e-6)
        s.update(t=t, over=over)
        rows.append(s)
        poses.append((t, root.copy(), q.copy()))

        out = pd.DataFrame([a9.iloc[A9_CROUCH_FRAME].values] * 5, columns=a9.columns)
        out[jc] = np.rad2deg(q)
        out[tc[2]] = root[2] * 100.0
        out[rc] = Rotation.from_quat(quat_seated[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        out.to_csv(f"/tmp/blend_{int(t * 100):03d}.csv", index=False, float_format="%.6f")

    print(f"{'blend':>7s}{'pelvis':>9s}{'lateral':>9s}{'stride':>8s}"
          f"{'margin':>9s}{'sole flat':>11s}{'over limit':>12s}")
    for s in rows:
        tag = " (seed crouch)" if s["t"] == 0 else (" (A9)" if s["t"] == 1 else "")
        print(f"{s['t'] * 100:6.0f}%{s['pelvis']:8.1f}c{s['lateral']:8.1f}c{s['stride']:7.1f}c"
              f"{s['margin']:+8.1f}c{s['flat']:10.1f}c{s['over']:11d}{tag}")
    print("\n  pelvis/lateral/stride/margin in cm; margin>0 = centre of mass inside the feet")


if __name__ == "__main__":
    main()
