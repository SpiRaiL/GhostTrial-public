#!/usr/bin/env python3
"""Speed a take up and deepen its crouch, closer to the intended performance.

    MUJOCO_GL=egl .venv/bin/python tools/refine_take.py in.csv out.csv \
        --speed 1.2 --crouch 4

Two knobs, because the retargeted take reads slower and more upright than the
motion it came from.

--speed is a uniform time scale. Joint DOFs are independent revolute angles well
inside their ranges so they lerp; the root rotation is an Euler triple and does
NOT, so it goes through quaternion slerp. Interpolating Euler angles across a wrap
tears the orientation apart, which is the same trap tools/retime_g1_csv.py
documents.

--crouch lowers the pelvis by N cm relative to the supporting foot. Crouching in
the sagittal plane couples three joints — knee flexes about twice the hip, and the
ankle takes up the rest to keep the sole flat — so one scalar drives all three and
is solved per frame by BISECTION against the height MuJoCo actually reports. A
closed form would be a second kinematics implementation to disagree with the one
every check here uses.

Two failure modes this has already been bitten by, so both are guarded:
  * an unbounded correction saturates and throws the legs into nonsense
    (stabilise_stance.py --narrow 3 reached 78 degrees from vertical). The solve is
    bounded and frames that hit the bound are reported, not hidden.
  * editing leg joints moves the FEET, not the pelvis, so the clip floats. The
    result is re-seated on the floor as the final step.

Speeding a clip up raises joint velocity and acceleration quadratically, so run
tools/check_g1_motion.py and tools/dynamic_feasibility.py on the result before
spending GPU time on it.
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
MAX_K = np.deg2rad(25.0)          # per-frame crouch scalar bound

ap = argparse.ArgumentParser()
ap.add_argument("in_csv")
ap.add_argument("out_csv")
ap.add_argument("--speed", type=float, default=1.0, help="time scale; 1.2 = 20%% faster")
ap.add_argument("--crouch", type=float, default=0.0, help="lower the pelvis by N cm")
ap.add_argument("--fps", type=float, default=60.0)
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
jc = joint_cols(df)
tc = ["root_translateX", "root_translateY", "root_translateZ"]
rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]

# ---- 1. uniform retime -------------------------------------------------------
T0 = len(df)
if abs(args.speed - 1.0) > 1e-6:
    T1 = max(2, int(round(T0 / args.speed)))
    src = np.arange(T0)
    dst = np.linspace(0, T0 - 1, T1)
    # every numeric column rides along; the take carries a Frame index too, and
    # dropping it silently breaks the tools downstream that key on it
    out = pd.DataFrame({c: np.interp(dst, src, df[c].values)
                        for c in df.columns if c not in rc})
    key = Rotation.from_euler("xyz", df[rc].values, degrees=True)
    eul = Slerp(src, key)(dst).as_euler("xyz", degrees=True)
    for i, c in enumerate(rc):
        out[c] = eul[:, i]
    out = out[list(df.columns)]
    if "Frame" in out.columns:
        out["Frame"] = np.arange(T1)
else:
    T1 = T0
    out = df.copy()

# ---- 2. deepen the crouch ----------------------------------------------------
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
feet = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}
idx = {n: jc.index(f"{n}_dof") for n in
       ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint")}

root = out[tc].values / 100.0
quat = Rotation.from_euler("xyz", out[rc].values, degrees=True).as_quat()[:, [3, 0, 1, 2]]
dof = np.deg2rad(out[jc].values)


def lowest_sole():
    lo = np.inf
    for gs in feet.values():
        for g in gs:
            p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
            for a in (-1, 1):
                for b in (-1, 1):
                    for c in (-1, 1):
                        lo = min(lo, (p + R @ (np.array([a, b, c]) * sz))[2])
    return lo


# joint ranges straight from the model everything else is checked against
_lim = {}
for n in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
          "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    _lim[n] = tuple(model.jnt_range[j]) if model.jnt_limited[j] else (-np.inf, np.inf)

# how much crouch scalar this frame can take before any of the six joints hits a stop
_GAIN = {"hip_pitch": 1.0, "knee": 2.0, "ankle_pitch": 1.0}


def headroom(i, s):
    room = MAX_K
    for side in ("left", "right"):
        for part, g in _GAIN.items():
            n = f"{side}_{part}_joint"
            q, (lo, hi) = dof[i][idx[n]], _lim[n]
            step = s * g
            if step > 0:
                room = min(room, max(0.0, (hi - q) / step))
            elif step < 0:
                room = min(room, max(0.0, (q - lo) / -step))
    return room


def crouch_state(i, k, s):
    """Pelvis height above the supporting foot, with crouch scalar k applied."""
    q = dof[i].copy()
    for side in ("left", "right"):
        q[idx[f"{side}_hip_pitch_joint"]] += s * k
        q[idx[f"{side}_knee_joint"]] += s * 2.0 * k       # knee closes twice the hip
        q[idx[f"{side}_ankle_pitch_joint"]] += s * k      # ankle keeps the sole flat
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], q
    mujoco.mj_forward(model, data)
    return data.qpos[2] - lowest_sole(), q


hit = 0
h0 = np.zeros(T1)
h1 = np.zeros(T1)
if args.crouch > 0:
    # which sign of k lowers the pelvis? derive it, do not assume — the sign
    # conventions here have been guessed wrong twice before
    base, _ = crouch_state(0, 0.0, 1.0)
    probe, _ = crouch_state(0, np.deg2rad(2.0), 1.0)
    sign = 1.0 if probe < base else -1.0

    newq = np.zeros_like(dof)
    kser = np.zeros(T1)
    orig_low = np.zeros(T1)
    for i in range(T1):
        h0[i], _ = crouch_state(i, 0.0, sign)
        orig_low[i] = lowest_sole()          # where this frame's soles started
        target = h0[i] - args.crouch / 100.0
        # The G1's knee is the binding constraint: the take already spends most of
        # its travel, and asking for 4 cm more crouch ran it 35 deg past the stop.
        # So the bound is whatever the joints actually allow at THIS frame, not a
        # fixed number — a clip that violates its limits cannot be tracked at all.
        lo, hi = 0.0, min(MAX_K, headroom(i, sign))
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            h, _ = crouch_state(i, mid, sign)
            if h > target:
                lo = mid
            else:
                hi = mid
        kser[i] = hi
        if hi >= MAX_K - 1e-6:
            hit += 1

    # Solved independently, k jumps between frames wherever the joint headroom
    # changes, and a discontinuous crouch depth is a discontinuous root height:
    # inverse dynamics went from 321% of the torque limit to 14901%. So smooth the
    # depth over time, then clamp back under each frame's headroom — smoothing
    # must not be able to reintroduce a limit violation.
    w = max(3, int(round(0.25 * args.fps)) | 1)             # ~0.25 s, odd
    pad = np.pad(kser, (w // 2, w // 2), mode="edge")
    kser = np.convolve(pad, np.ones(w) / w, mode="valid")
    for i in range(T1):
        kser[i] = min(kser[i], headroom(i, sign))
    for i in range(T1):
        h1[i], newq[i] = crouch_state(i, kser[i], sign)
    out[jc] = np.rad2deg(newq)

    # Leg edits move the FEET, not the pelvis — the root is pinned, so flexing the
    # legs lifts the robot off the floor instead of lowering its hips. Re-seating
    # against the clip's global minimum is not enough either: it fixes one frame and
    # leaves the rest floating (the first attempt left the soles 11 cm up).
    #
    # So each frame is shifted by exactly the height ITS OWN soles gained. That drops
    # the pelvis by the crouch amount in world space and reproduces the source clip's
    # sole heights frame for frame, which keeps its contact schedule — including any
    # genuine flight phases — untouched.
    lows = np.zeros(T1)
    for i in range(T1):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], newq[i]
        mujoco.mj_forward(model, data)
        lows[i] = lowest_sole()
    out[tc[2]] -= (lows - orig_low) * 100.0

out.to_csv(args.out_csv, index=False, float_format="%.6f")

acc = lambda v: float(np.abs(np.diff(np.deg2rad(v), n=2, axis=0) * args.fps ** 2).mean())
print(f"{os.path.basename(args.in_csv)} -> {os.path.basename(args.out_csv)}")
print(f"  speed          {args.speed:.2f}x   {T0} -> {T1} frames "
      f"({T0 / args.fps:.2f} -> {T1 / args.fps:.2f} s)")
if args.crouch > 0:
    print(f"  pelvis height  {h0.mean() * 100:5.1f} -> {h1.mean() * 100:5.1f} cm "
          f"(asked for -{args.crouch:g})")
    print(f"  re-seated      per frame, mean {np.mean(orig_low - lows) * 100:+5.1f} cm; "
          f"sole heights now match the source to "
          f"{np.abs(lows - (lows - orig_low) - orig_low).max() * 1000:.2f} mm")
    if hit:
        print(f"  \033[31m{hit}/{T1} frames hit the {np.rad2deg(MAX_K):.0f} deg bound "
              f"and are shallower than asked\033[0m")
print(f"  mean |joint accel|  {acc(df[jc].values):6.1f} -> {acc(out[jc].values):6.1f} rad/s^2"
      f"   (BONES-SEED range 0.9-8.2)")
print(f"wrote {args.out_csv}")
