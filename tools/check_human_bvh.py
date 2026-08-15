#!/usr/bin/env python3
"""Is the reconstructed human actually a human? Measure it before retargeting.

    .venv/bin/python tools/check_human_bvh.py

A motion can look plausible in a viewport and still be unusable: limbs the wrong
length for the performer, feet sliding or sinking through the floor, or per-frame
jitter that a tracking controller will chase. Those are the failures that survive
retargeting and get blamed on the robot later, so they get checked here, on the
human, where they are still fixable by reshooting.

Generic BVH parser + FK — no GEM-X dependency, so this also runs against anything
else we might compare against.
"""

import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BVHDIR = os.path.join(REPO, "data", "human_bvh")


def parse_bvh(path):
    txt = open(path).read()
    head, _, motion = txt.partition("MOTION")
    names, offsets, parents, channels = [], [], [], []
    stack = []
    tok = re.findall(r"\S+", head)
    i = 0
    while i < len(tok):
        t = tok[i]
        if t in ("ROOT", "JOINT", "End"):
            nm = tok[i + 1] if t != "End" else names[stack[-1]] + "_end"
            parents.append(stack[-1] if stack else -1)
            names.append(nm)
            offsets.append([0, 0, 0])
            channels.append([])
            stack.append(len(names) - 1)
            i += 2
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(x) for x in tok[i + 1:i + 4]]
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            channels[stack[-1]] = tok[i + 2:i + 2 + n]
            i += 2 + n
        elif t == "}":
            stack.pop()
            i += 1
        else:
            i += 1

    lines = [l for l in motion.strip().splitlines()]
    nframes = int(lines[0].split(":")[1])
    ftime = float(lines[1].split(":")[1])
    data = np.array([[float(x) for x in l.split()] for l in lines[2:2 + nframes]])
    return (names, np.array(offsets), np.array(parents), channels, data,
            nframes, 1.0 / ftime)


def rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    z = np.zeros_like(c)
    o = np.ones_like(c)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.array([[np.broadcast_to(e, c.shape) for e in r] for r in m]).transpose(2, 0, 1)


def fk(names, offsets, parents, channels, data, scale=0.01, return_rots=False):
    """World-space joint positions in metres, (T, J, 3).

    SOMA's exporter gives EVERY joint translation channels, not just the root, and
    the actual root motion rides on Hips rather than Root. Per the BVH convention
    those channels replace the static OFFSET, so use them wherever they exist —
    read only the offsets and the figure never moves.
    """
    T, J = len(data), len(names)
    pos = np.zeros((T, J, 3))
    rots = np.zeros((T, J, 3, 3))
    col = 0
    for j in range(J):
        R = np.tile(np.eye(3), (T, 1, 1))
        loc = np.tile(offsets[j].astype(float), (T, 1))
        has_pos = False
        for c in channels[j]:
            v = data[:, col]
            col += 1
            if c.endswith("position"):
                if not has_pos:
                    loc = np.zeros((T, 3))
                    has_pos = True
                loc[:, "XYZ".index(c[0])] = v
            else:
                R = R @ rot(c[0], v)
        p = parents[j]
        if p < 0:
            rots[:, j] = R
            pos[:, j] = loc
        else:
            rots[:, j] = rots[:, p] @ R
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", rots[:, p], loc)
    return (pos * scale, rots) if return_rots else pos * scale


def report(path):
    names, offsets, parents, channels, data, T, fps = parse_bvh(path)
    pos = fk(names, offsets, parents, channels, data)
    idx = {n: i for i, n in enumerate(names)}
    print(f"\n\033[1m{os.path.basename(path)}\033[0m  {T} frames @ {fps:.0f} fps  "
          f"({T / fps:.1f}s), {len(names)} nodes")

    # a child's OFFSET is the parent bone's length vector; BVH is in centimetres
    def chain(*js):
        return sum(float(np.linalg.norm(offsets[idx[j]])) * 0.01 for j in js if j in idx)

    # ── skeleton proportions ────────────────────────────────────────────────
    feet = [idx[n] for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")
            if n in idx]
    floor = float(np.percentile(pos[:, feet, 1].min(axis=1), 2))
    # stature off the skull top, NOT max-over-all-joints — the uppercut puts a wrist
    # well above the head and that would read as a 2 m performer
    height = float(np.percentile(pos[:, idx["HeadEnd"], 1] - floor, 98))
    thigh, shin = chain("LeftShin"), chain("LeftFoot")
    upper, fore = chain("LeftForeArm"), chain("LeftHand")
    print(f"  stature              {height:6.3f} m  (skull top over floor, p98)")
    print(f"  leg (thigh+shin)     {thigh + shin:6.3f} m   "
          f"{(thigh + shin) / height * 100:4.1f}% of stature (human ~48-52%)")
    print(f"  arm (upper+fore)     {upper + fore:6.3f} m   "
          f"{(upper + fore) / height * 100:4.1f}% of stature (human ~31-34%)")

    # left/right symmetry — the model solves one identity, so asymmetry is a red flag
    asym = []
    for l in names:
        if not l.startswith("Left"):
            continue
        r = "Right" + l[4:]
        if r in idx:
            a, b = np.linalg.norm(offsets[idx[l]]), np.linalg.norm(offsets[idx[r]])
            if max(a, b) > 1e-4:
                asym.append(abs(a - b) / max(a, b))
    print(f"  L/R asymmetry        {np.max(asym) * 100:5.2f}% max, "
          f"{np.mean(asym) * 100:4.2f}% mean")

    # ── ground contact ──────────────────────────────────────────────────────
    fy = pos[:, feet, 1]
    pen = np.clip(floor - fy.min(axis=1), 0, None)
    print(f"  floor level          {floor:+.3f} m   "
          f"penetration max {pen.max() * 1000:.0f} mm, mean {pen.mean() * 1000:.1f} mm")

    # foot slide while planted: horizontal speed of a foot that is on the ground
    slide = []
    for f in feet[:2]:
        h = pos[:, f, 1] - floor
        planted = h < 0.05
        v = np.linalg.norm(np.diff(pos[:, f, [0, 2]], axis=0), axis=1) * fps
        m = planted[:-1] & planted[1:]
        if m.sum():
            slide.append(float(np.percentile(v[m], 90)))
    print(f"  planted-foot slide   {max(slide):5.3f} m/s (p90)   "
          f"{'OK' if max(slide) < 0.20 else 'HIGH'}")

    # ── crouch depth, the thing we asked him to exaggerate ──────────────────
    # Root sits at the origin for the whole clip; Hips carries the translation
    root = pos[:, idx["Hips"], 1] - floor
    print(f"  pelvis height        {root.max():.3f} -> {root.min():.3f} m   "
          f"drop {(root.max() - root.min()) * 100:.1f} cm")

    # ── jitter: third derivative, the thing SONIC will chase ────────────────
    j3 = np.abs(np.diff(pos, n=3, axis=0)).mean() * fps ** 3
    wrist = [idx[n] for n in ("LeftHand", "RightHand") if n in idx]
    jw = np.abs(np.diff(pos[:, wrist], n=3, axis=0)).mean() * fps ** 3
    print(f"  jerk (all joints)    {j3:7.1f} m/s^3   wrists {jw:7.1f} m/s^3")
    return dict(height=height, floor=floor, pen=pen.max(), slide=max(slide),
                drop=root.max() - root.min(), jerk=j3, jerk_wrist=jw,
                asym=float(np.max(asym)), T=T, fps=fps,
                leg_pct=(thigh + shin) / height * 100,
                arm_pct=(upper + fore) / height * 100)


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(BVHDIR) if f.endswith(".bvh"))
    out = {f: report(os.path.join(BVHDIR, f)) for f in files}
    print("\nmeasured performer height: "
          f"{np.mean([v['height'] for v in out.values()]):.3f} m "
          f"(spread {np.ptp([v['height'] for v in out.values()]) * 1000:.0f} mm across takes)")
