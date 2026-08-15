#!/usr/bin/env python3
"""Rebuild the take from key poses that have each been re-seated on the floor.

    MUJOCO_GL=egl .venv/bin/python tools/retarget_via_keys.py --blend 0.2

The procedure David picked, applied along the whole clip instead of one frame:

  1. find key frames — the extremes of the performance, where the pose is doing
     something, plus enough fill that no gap exceeds FILL_S seconds
  2. at each key, blend the LOWER BODY toward the BONES-SEED crouch, keep the arms
     and head from the capture, and seat the whole robot on the floor by rotating and
     translating it rigidly — never by bending the legs, which would rewrite the pose
     being chosen
  3. tween between the seated keys, and re-seat every tweened frame too, so the feet
     stay planted through the interpolation rather than only at the keys

Two things are deliberately NOT uniform along the clip:

  * the waist. At the crouch we want the seed reference's bent back, but forcing it
    everywhere would freeze the torso through the uppercut. Its weight follows how
    crouched each frame is — full at the deepest, none at the tallest.
  * the facing. Only the LEAN blends. The seed crouch faces -83.6 degrees and the
    take faces +90.4, so blending whole root orientations swings the character 174
    degrees.
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_columns import joint_cols  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
FILL_S = 0.35                 # never leave a gap longer than this between keys
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST = ("waist_yaw", "waist_roll", "waist_pitch")

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
FEET = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}


def corners(side):
    pts = []
    for g in FEET[side]:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def seat(root, quat, q, iters=6):
    """Rotate and translate the WHOLE robot until both soles lie on the floor.

    The joints are the pose; the root is free. Averages the two feet's own up-axes
    rather than fitting a plane to the contact points — that fit degenerates when the
    points are near-collinear and once rotated the robot face-down.
    """
    root = np.asarray(root, dtype=float).copy()
    data.qpos[7:] = q
    for _ in range(iters):
        data.qpos[:3], data.qpos[3:7] = root, quat
        mujoco.mj_forward(model, data)
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
        axis = np.cross(n, [0.0, 0.0, 1.0])
        sa = np.linalg.norm(axis)
        if sa < 1e-9:
            break
        angle = np.arctan2(sa, float(np.dot(n, [0.0, 0.0, 1.0])))
        if abs(angle) < 1e-5 or abs(angle) > np.deg2rad(60.0):
            break
        R = Rotation.from_rotvec(axis / sa * angle)
        c = np.mean(cs, axis=0)
        root = R.apply(root - c) + c
        quat = (R * Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    data.qpos[:3], data.qpos[3:7] = root, quat
    mujoco.mj_forward(model, data)
    root[2] -= min(corners("left")[:, 2].min(), corners("right")[:, 2].min())
    return root, quat


def find_keys(df, jc, fps):
    """Frames where the performance turns, plus fill so nothing is left to guesswork."""
    q = np.deg2rad(df[jc].values)
    sig = np.column_stack([
        q[:, [i for i, c in enumerate(jc) if "hip_pitch" in c]].mean(axis=1),
        q[:, [i for i, c in enumerate(jc) if "knee" in c]].mean(axis=1),
        q[:, [i for i, c in enumerate(jc) if "shoulder_pitch" in c]].mean(axis=1),
        q[:, [i for i, c in enumerate(jc) if "elbow" in c]].mean(axis=1),
    ])
    sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-9)
    keys = {0, len(df) - 1}
    for k in range(sig.shape[1]):
        v = sig[:, k]
        d = np.diff(v)
        turn = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
        for t in turn:                       # only turns that actually mean something
            lo = max(0, t - int(0.15 * fps))
            hi = min(len(v), t + int(0.15 * fps))
            if v[lo:hi].max() - v[lo:hi].min() > 0.35:
                keys.add(int(t))
    keys = sorted(keys)
    filled, gap = [keys[0]], int(FILL_S * fps)
    for a, b in zip(keys, keys[1:]):
        n = int((b - a) // gap)
        filled += [a + int((b - a) * (j + 1) / (n + 1)) for j in range(n)]
        filled.append(b)
    return sorted(set(filled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gemx_g1_retimed/A9_smooth.csv")
    ap.add_argument("--ref", default="data/csv_frozen/REF_crouch.csv")
    ap.add_argument("--blend", type=float, default=0.2,
                    help="0 = seed crouch legs, 1 = the take's own legs")
    ap.add_argument("--out", default="data/gemx_g1_retimed/T5_keyed.csv")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--video-fps", type=float, default=30.0,
                    help="rate the review videos play at; --at times are read off those")
    ap.add_argument("--at", default="17,20,23",
                    help="seconds (in the review video) of blend-in, full crouch, blend-out")
    args = ap.parse_args()

    t = args.blend
    df = pd.read_csv(args.src)
    ref = pd.read_csv(args.ref)
    jc = joint_cols(df)
    idx = {c[:-4]: i for i, c in enumerate(jc)}
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]

    Q = np.deg2rad(df[jc].values)
    ROOT = df[tc].values / 100.0
    ROT = Rotation.from_euler("xyz", df[rc].values, degrees=True)
    qr = np.deg2rad(ref[joint_cols(ref)].values[0])
    rot_r = Rotation.from_euler("xyz", ref[rc].values[0], degrees=True)
    tilt_r = Rotation.from_euler("z", rot_r.as_euler("zyx")[0]).inv() * rot_r

    leg_cols = [idx[f"{s}_{p}_joint"] for s in ("left", "right") for p in LEG]
    # As an OFFSET, not a replacement. Blending each frame toward a single static
    # reference at t=0.2 makes every frame 80% the SAME pose: the first version of
    # this collapsed knee range from 74 degrees to 16 and froze the legs. Instead take
    # the shift the approved crouch frame needed and apply that shift everywhere, so
    # the take's own leg motion rides on top of a seed-like posture. At the crouch
    # frame this reproduces exactly the pose that was picked.
    A9_CROUCH_FRAME = 609
    leg_delta = np.zeros_like(qr)
    for c in leg_cols:
        leg_delta[c] = qr[c] - Q[A9_CROUCH_FRAME][c]

    # Blend schedule over time, not a constant. The crouch correction is wanted only
    # around the coil; everywhere else the take should be untouched. Times are given
    # in REVIEW-VIDEO seconds because that is what gets watched — the clip is 983
    # frames, which is 16.4 s of motion at 60 fps but 32.8 s as rendered at 30.
    t_in, t_mid, t_out = (float(x) for x in args.at.split(","))
    f_in, f_mid, f_out = (int(round(x * args.video_fps)) for x in (t_in, t_mid, t_out))

    def smoothstep(a, b, x):
        if b == a:
            return 0.0
        u = np.clip((x - a) / (b - a), 0.0, 1.0)
        return u * u * (3 - 2 * u)          # zero slope at both ends: no velocity step

    def blend_at(f):
        """1.0 = the take untouched, args.blend at the bottom of the coil."""
        if f <= f_in or f >= f_out:
            return 1.0
        if f <= f_mid:
            return 1.0 - (1.0 - t) * smoothstep(f_in, f_mid, f)
        return t + (1.0 - t) * smoothstep(f_mid, f_out, f)
    waist_cols = [idx[f"{w}_joint"] for w in WAIST]

    # how crouched is each frame, 0..1 — drives how much of the seed's back to use
    pelvis = np.zeros(len(df))
    for i in range(len(df)):
        data.qpos[:3], data.qpos[3:7] = ROOT[i], ROT[i].as_quat()[[3, 0, 1, 2]]
        data.qpos[7:] = Q[i]
        mujoco.mj_forward(model, data)
        pelvis[i] = data.qpos[2] - min(corners("left")[:, 2].min(), corners("right")[:, 2].min())
    crouchness = np.clip((np.percentile(pelvis, 95) - pelvis)
                         / max(np.percentile(pelvis, 95) - np.percentile(pelvis, 5), 1e-6), 0, 1)

    keys = find_keys(df, jc, args.fps)
    yaw_c = Rotation.from_euler("z", ROT[609].as_euler("zyx")[0])
    tilt_c = yaw_c.inv() * ROT[609]
    tilt_full = tilt_r * tilt_c.inv()
    kq, kroot, krot = [], [], []
    for i in keys:
        ti = blend_at(i)
        q = Q[i].copy()
        w = (1.0 - ti) * crouchness[i]       # seed's back only where the coil is
        for c in waist_cols:
            q[c] = qr[c] * w + Q[i][c] * (1 - w)
        for c in leg_cols:
            # clamp to the joint's real range: a constant offset added to a moving
            # joint will push it past its stop somewhere in the clip, and it did —
            # violations went 201 to 381 before this
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jc[c][:-4])
            lo, hi = (model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf))
            q[c] = float(np.clip(Q[i][c] + leg_delta[c] * (1.0 - ti), lo, hi))
        yaw_i = Rotation.from_euler("z", ROT[i].as_euler("zyx")[0])
        tilt_i = yaw_i.inv() * ROT[i]
        tilt = Slerp([0.0, 1.0], Rotation.concatenate(
            [tilt_full, Rotation.identity()]))([ti])[0] * tilt_i
        root, quat = seat(ROOT[i], (yaw_i * tilt).as_quat()[[3, 0, 1, 2]], q)
        kq.append(q)
        kroot.append(root)
        krot.append(Rotation.from_quat(quat[[1, 2, 3, 0]]))

    # tween: joints lerp, root position lerps, root rotation slerps
    kq = np.array(kq)
    kroot = np.array(kroot)
    ks = np.array(keys, dtype=float)
    allf = np.arange(len(df), dtype=float)
    outq = np.column_stack([np.interp(allf, ks, kq[:, j]) for j in range(kq.shape[1])])
    outroot = np.column_stack([np.interp(allf, ks, kroot[:, j]) for j in range(3)])
    outrot = Slerp(ks, Rotation.concatenate(krot))(np.clip(allf, ks[0], ks[-1]))

    # re-seat every tweened frame: interpolation between two seated poses is not itself seated
    lifted = 0
    for i in range(len(df)):
        r, qt = seat(outroot[i], outrot[i].as_quat()[[3, 0, 1, 2]], outq[i], iters=3)
        if abs(r[2] - outroot[i][2]) > 0.005:
            lifted += 1
        outroot[i] = r
        outrot[i] = Rotation.from_quat(qt[[1, 2, 3, 0]])

    out = df.copy()
    out[jc] = np.rad2deg(outq)
    out[tc] = outroot * 100.0
    out[rc] = outrot.as_euler("xyz", degrees=True)
    out.to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(keys)} key poses over {len(df)} frames "
          f"(every {len(df) / max(len(keys) - 1, 1) / args.fps:.2f} s on average)")
    print(f"  untouched until {t_in:g}s, {t:.0%} of the take at {t_mid:g}s, "
          f"back to untouched by {t_out:g}s  (review-video seconds at {args.video_fps:g} fps)")
    print(f"  = frames {f_in} / {f_mid} / {f_out} of {len(df)}; "
          f"deepest crouch in the take is frame {A9_CROUCH_FRAME}")
    print(f"  {lifted} tweened frames needed re-seating beyond 5 mm")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
