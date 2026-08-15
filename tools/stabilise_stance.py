#!/usr/bin/env python3
"""Take A5 — a stance the G1 can actually balance in.

    .venv/bin/python tools/stabilise_stance.py --narrow 0.5 --crouch 0.7 --slow 1.25

The policy shuffles. Its feet travel 1.05-1.36 m against the reference's 0.24-0.27 m,
sliding at over 1 m/s, because it is stepping constantly to stay upright rather
than performing the move.

The cause is upstream of the policy. The performer stands in a wide fighting
stance with his toes well turned out — hip yaw -38.7 deg, hip roll splayed — and
the G1 cannot hold that flat-footed: ankle roll saturates at its +/-15 deg stop
even in the IDLE frames, before the move starts. With ankle roll pinned there is
no lateral authority left, so the only way the policy can balance is to take a
step. No amount of training fixes that; the reference has to ask for something the
robot can stand in.

So this narrows the stance, brings the toes in, shallows the crouch and slows the
whole thing down — the "under-exaggerate it" the brief allows for. Edits are made
on the HUMAN skeleton and re-retargeted, so the retargeter's own IK re-places the
feet and re-solves the ankles rather than us forcing joint angles that then do not
touch the floor.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT = os.path.join(REPO, "data", "human_bvh_edited")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(EDIT, "A4_natural_speed_human.bvh"))
ap.add_argument("--out", default=os.path.join(EDIT, "A5_stable_stance_human.bvh"))
ap.add_argument("--narrow", type=float, default=0.5,
                help="fraction of the leg's lateral splay and toe-out to remove")
ap.add_argument("--crouch", type=float, default=0.7,
                help="fraction of the pelvis drop to keep (1.0 = unchanged)")
ap.add_argument("--slow", type=float, default=1.25, help="playback stretch factor")
ap.add_argument("--toein", type=float, default=0.0,
                help="fraction of each foot's toe-out to remove. The performer stands "
                     "with his feet turned well out (left -25 deg, right +46 deg from "
                     "body-forward, peaking at 82). The G1 cannot keep a foot flat at "
                     "that yaw within its +/-15 deg of ankle roll, so the ankle "
                     "saturates and the policy loses the authority it balances with.")
args = ap.parse_args()

names, off, par, channels, data, T, fps = parse_bvh(args.src)
idx = {n: i for i, n in enumerate(names)}
pos, rots = fk(names, off, par, channels, data, return_rots=True)
UP = np.array([0.0, 1.0, 0.0])

col, cols = 0, {}
for j, n in enumerate(names):
    cols[n] = col
    col += len(channels[j])


def rot_cols_for(bone):
    ks = [k for k, c in enumerate(channels[idx[bone]]) if c.endswith("rotation")]
    seq = "".join(channels[idx[bone]][k][0] for k in ks)
    return [cols[bone] + k for k in ks], seq


out = data.copy()

# ── 1. narrow the stance: swing each thigh toward the midline ────────────────
# Rotating the thigh about the body-forward axis changes how far the leg splays
# sideways; the sign that pulls it inward is derived, not assumed.
feet_i = [idx["LeftToeBase"], idx["RightToeBase"]]
mid = pos[:, idx["Hips"]]
for side, other in (("Left", "Right"), ("Right", "Left")):
    bone = side + "Leg"
    rc, seq = rot_cols_for(bone)
    hip = pos[:, idx[bone]]
    foot = pos[:, idx[side + "Foot"]]
    for f in range(T):
        lat = pos[f, idx["RightLeg"]] - pos[f, idx["LeftLeg"]]
        lat[1] = 0.0
        if np.linalg.norm(lat) < 1e-6:
            continue
        fwd = np.cross(UP, lat / np.linalg.norm(lat))
        fwd /= np.linalg.norm(fwd)
        # how far out to the side is this foot from the pelvis, in the lateral dir?
        latn = lat / np.linalg.norm(lat)
        sway = float(np.dot(foot[f] - mid[f], latn))
        leg = np.linalg.norm(foot[f] - hip[f])
        if leg < 1e-6:
            continue
        # angle to remove, about the forward axis, to pull the foot in by `narrow`
        ang = np.arcsin(np.clip(sway * args.narrow / leg, -0.9, 0.9))
        sgn = -1.0 if side == "Right" else 1.0
        R = Rotation.from_rotvec(fwd * (sgn * ang)).as_matrix()
        p = par[idx[bone]]
        Rp = rots[f, p]
        new = Rp.T @ R @ Rp @ (Rp.T @ rots[f, idx[bone]])
        out[f, rc] = Rotation.from_matrix(new).as_euler(seq, degrees=True)

# ── 1b. bring the toes in: yaw each leg about vertical ───────────────────────
if abs(args.toein) > 1e-6:
    for side in ("Left", "Right"):
        bone = side + "Leg"
        rc, seq = rot_cols_for(bone)
        for f in range(T):
            lat = pos[f, idx["RightLeg"]] - pos[f, idx["LeftLeg"]]
            lat[1] = 0.0
            if np.linalg.norm(lat) < 1e-6:
                continue
            latn = lat / np.linalg.norm(lat)
            fwd = np.cross(UP, latn)
            toe = pos[f, idx[side + "ToeBase"]] - pos[f, idx[side + "Foot"]]
            toe[1] = 0.0
            if np.linalg.norm(toe) < 1e-6:
                continue
            toe /= np.linalg.norm(toe)
            yaw = np.arctan2(float(np.dot(toe, latn)), float(np.dot(toe, fwd)))
            R = Rotation.from_rotvec(UP * (args.toein * yaw)).as_matrix()
            p_ = par[idx[bone]]
            Rp = rots[f, p_]
            new = Rp.T @ R @ Rp @ (Rp.T @ rots[f, idx[bone]])
            out[f, rc] = Rotation.from_matrix(new).as_euler(seq, degrees=True)

# ── 2. shallower crouch: STRAIGHTEN THE LEGS ─────────────────────────────────
# Not by lifting the pelvis. Moving the Hips channel translates the whole body,
# feet included, so the pelvis-to-foot distance — which is what "crouch" means —
# never changes, and the retargeter's foot stabiliser puts it straight back. The
# crouch lives in the leg JOINTS, so blend those toward their idle pose instead.
if abs(args.crouch - 1.0) > 1e-6:
    for bone in ("LeftLeg", "LeftShin", "LeftFoot",
                 "RightLeg", "RightShin", "RightFoot"):
        if bone not in idx:
            continue
        rc, seq = rot_cols_for(bone)
        ang = out[:, rc]
        R = Rotation.from_euler(seq, ang, degrees=True)
        idle = Rotation.from_euler(seq, ang[:40].mean(axis=0), degrees=True)
        keyR = Rotation.concatenate([idle, R[0]])
        blended = []
        for f in range(len(ang)):
            pair = Rotation.concatenate([idle, R[f]])
            s = Slerp([0.0, 1.0], pair)
            blended.append(s([args.crouch])[0].as_euler(seq, degrees=True))
        out[:, rc] = np.array(blended)

# ── 3. slow it down ──────────────────────────────────────────────────────────
if abs(args.slow - 1.0) > 1e-6:
    n = int(round(T * args.slow))
    src_f = np.arange(T)
    tgt = np.linspace(0, T - 1, n)
    slowed = np.zeros((n, out.shape[1]))
    c = 0
    for j in range(len(names)):
        ch = channels[j]
        if not ch:
            continue
        pk = [k for k, x in enumerate(ch) if x.endswith("position")]
        rk = [k for k, x in enumerate(ch) if x.endswith("rotation")]
        for k in pk:
            slowed[:, c + k] = np.interp(tgt, src_f, out[:, c + k])
        if rk:
            seq = "".join(ch[k][0] for k in rk)
            R = Rotation.from_euler(seq, out[:, [c + k for k in rk]], degrees=True)
            e = Slerp(src_f, R)(tgt).as_euler(seq, degrees=True)
            for mI, k in enumerate(rk):
                slowed[:, c + k] = e[:, mI]
        c += len(ch)
    out = slowed

# Re-seat on the floor. Editing LEG joints moves the feet, not the pelvis, because
# Hips is the root of the hierarchy — so shallowing the crouch lifts the whole
# figure. Unregrounded, these takes floated 23-41 cm off the ground.
tmp_pos = None
_names, _off, _par, _ch, _d, _T, _fps = names, off, par, channels, out, len(out), fps
_pos = fk(_names, _off, _par, _ch, _d)
_feet = [idx[n] for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")
         if n in idx]
_low = _pos[:, _feet, 1].min(axis=1)
out[:, cols["Hips"] + 1] -= (_low - np.percentile(_low, 2)) * 100.0

txt = open(args.src).read()
lines = ["MOTION", f"Frames: {len(out)}", f"Frame Time: {1.0 / fps:.6f}"]
lines += [" ".join(f"{x:.6f}" for x in r) for r in out]
open(args.out, "w").write(txt[:txt.index("MOTION")] + "\n".join(lines) + "\n")

n2, o2, p2, c2, d2, T2, _ = parse_bvh(args.out)
pos2 = fk(n2, o2, p2, c2, d2)
fl = [idx["LeftToeBase"], idx["RightToeBase"]]
w0 = np.linalg.norm((pos[:, fl[0]] - pos[:, fl[1]])[:, [0, 2]], axis=1)
w1 = np.linalg.norm((pos2[:, fl[0]] - pos2[:, fl[1]])[:, [0, 2]], axis=1)
fl0 = float(np.percentile(pos[:, feet_i, 1].min(axis=1), 2))
fl1 = float(np.percentile(pos2[:, feet_i, 1].min(axis=1), 2))
print(f"narrow {args.narrow:g}  crouch {args.crouch:g}  slow {args.slow:g}")
print(f"  stance width : {w0.mean() * 100:.1f} -> {w1.mean() * 100:.1f} cm")
# pelvis height ABOVE THE FEET, per frame. Hips is the hierarchy root and is never
# edited, so measuring it against a fixed floor always reports "unchanged" — the
# crouch shows up in where the feet end up relative to it.
h0 = pos[:, idx['Hips'], 1] - pos[:, feet_i, 1].min(axis=1)
h1 = pos2[:, idx['Hips'], 1] - pos2[:, feet_i, 1].min(axis=1)
print(f"  pelvis over feet: {h0.max()*100:.0f}->{h0.min()*100:.0f} cm (drop {np.ptp(h0)*100:.1f})"
      f"   becomes {h1.max()*100:.0f}->{h1.min()*100:.0f} cm (drop {np.ptp(h1)*100:.1f})")
print(f"  frames       : {T} -> {len(out)}  ({T / fps:.1f}s -> {len(out) / fps:.1f}s)")
print(f"wrote {args.out}")
