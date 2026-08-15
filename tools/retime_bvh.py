#!/usr/bin/env python3
"""Trim and locally speed up a BVH, without touching the poses.

    .venv/bin/python tools/retime_bvh.py            # all three takes, edited copies

Two edits are wanted on the solved human motion:

  * the clip opens on ~1 s of the performer holding a raised arm before he settles,
    so it should start at the idle stance instead;
  * the uppercut swing is slower than the move should read, so that span alone runs
    at double speed while the rest of the phrase keeps its original timing.

Both are pure time warps — every pose is preserved exactly, only *when* it happens
changes. Sampling between frames means interpolating rotations, and Euler channels
cannot be lerped safely (they wrap, and gimbal-adjacent frames blow up), so
rotations go through quaternion slerp and come back out in the file's own channel
order. Translations lerp.

The speed window is given in the frame numbers you read off Blender's timeline
(1-based, on the untrimmed clip), because that is where you saw the problem.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import BVHDIR, fk, parse_bvh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "data", "human_bvh_edited")

SWING_SPEED = 2.0     # the uppercut swing, doubled
RAMP = 10             # frames blended in/out of the fast span, so there is no
                      # velocity step — a hard cut would show as a jerk and would
                      # be penalised by SONIC's smoothness term


def find_stance(path):
    """The stillest frame in the opening idle — where the clip should start.

    tools/cut_full_phrase.py already cuts each take to open on a settled stance
    before the wind-up, so this only tidies the first few frames. It deliberately
    searches the START of the clip: an earlier version searched backwards from the
    crouch, which on these clips lands *after* the spear throw and silently threw
    the first half of the move away.
    """
    names, off, par, ch, data, T, fps = parse_bvh(path)
    idx = {n: i for i, n in enumerate(names)}
    pos = fk(names, off, par, ch, data)
    feet = [idx["LeftToeBase"], idx["RightToeBase"]]
    floor = float(np.percentile(pos[:, feet, 1].min(axis=1), 2))
    pelvis = pos[:, idx["Hips"], 1] - floor
    crouch = int(np.argmin(pelvis))

    # mean joint speed, smoothed over a third of a second
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(axis=1)
    speed = np.convolve(speed, np.ones(20) / 20, mode="same")
    hi = min(90, max(20, crouch // 3))          # the opening idle, never the coil
    return int(np.argmin(speed[:hi])), crouch, pelvis, pos, idx, floor


# The swing window, expressed as a fraction of each take's own crouch -> uppercut
# peak. The numbers come from take A's window read off the Blender timeline by eye
# (frames 320-400 on the earlier, shorter cut, where the crouch sat at 289 and the
# peak at 427). Held as a fraction rather than frame numbers so it survives the
# clips being re-cut, which they since were.
#
# Deliberately not auto-detected: the punching hand keeps creeping upward for ~65
# frames after the swing visually ends, so a 95%-of-peak threshold stops at 363,
# well short of the follow-through, while argmax runs to 428, deep into the hold.
# The eye is the right instrument for "where does the swing look like it is".
SWING_FRAC = ((320 - 1 - 289) / (427 - 289), (400 - 1 - 289) / (427 - 289))


def find_swing(pos, idx, floor, crouch):
    """The swing span, as the same fraction of crouch->peak in every take."""
    lift = {h: pos[:, idx[h], 1] - floor for h in ("LeftHand", "RightHand")}
    hand = max(lift, key=lambda h: lift[h][crouch:].max())
    peak = crouch + int(np.argmax(lift[hand][crouch:]))
    f0, f1 = SWING_FRAC
    span = peak - crouch
    return crouch + int(f0 * span), crouch + int(f1 * span), peak


def rate(t, a, b, speed, ramp):
    """Playback rate at input frame t — 1x outside, `speed` inside, ramped at the edges."""
    if t <= a - ramp or t >= b + ramp:
        return 1.0
    if a <= t <= b:
        return speed
    if t < a:
        return 1.0 + (speed - 1.0) * (t - (a - ramp)) / ramp
    return 1.0 + (speed - 1.0) * ((b + ramp) - t) / ramp


def retime(src, dst, trim, a, b, speed=SWING_SPEED, ramp=RAMP):
    names, off, par, channels, data, T, fps = parse_bvh(src)
    txt = open(src).read()
    hierarchy = txt[:txt.index("MOTION")]

    # walk the input timeline, stepping faster inside the window
    times, t = [], float(trim)
    while t <= T - 1:
        times.append(t)
        t += rate(t, a, b, speed, ramp)
    times = np.array(times)

    out = np.zeros((len(times), data.shape[1]))
    src_f = np.arange(T)
    col = 0
    for j in range(len(names)):
        ch = channels[j]
        if not ch:
            continue
        pos_cols = [k for k, c in enumerate(ch) if c.endswith("position")]
        rot_cols = [k for k, c in enumerate(ch) if c.endswith("rotation")]
        for k in pos_cols:
            out[:, col + k] = np.interp(times, src_f, data[:, col + k])
        if rot_cols:
            seq = "".join(ch[k][0] for k in rot_cols)          # e.g. "ZYX"
            ang = data[:, [col + k for k in rot_cols]]
            rots = Rotation.from_euler(seq, ang, degrees=True)
            interp = Slerp(src_f, rots)(times).as_euler(seq, degrees=True)
            for m, k in enumerate(rot_cols):
                out[:, col + k] = interp[:, m]
        col += len(ch)

    lines = ["MOTION", f"Frames: {len(times)}", f"Frame Time: {1.0 / fps:.6f}"]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in out]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(hierarchy + "\n".join(lines) + "\n")
    return T, len(times)


if __name__ == "__main__":
    for fn in sorted(f for f in os.listdir(BVHDIR) if f.endswith(".bvh")):
        src = os.path.join(BVHDIR, fn)
        stance, crouch, pelvis, pos, idx, floor = find_stance(src)
        a, b, peak = find_swing(pos, idx, floor, crouch)
        T, n = retime(src, os.path.join(OUTDIR, fn), stance, a, b)
        print(f"{fn[0]}  idle from f{stance + 1}  ·  crouch f{crouch + 1}, "
              f"uppercut peak f{peak + 1}  ·  swing f{a + 1}-{b + 1} at "
              f"{SWING_SPEED:g}x  ·  {T} -> {n} frames "
              f"({T / 60:.1f}s -> {n / 60:.1f}s)")
    print(f"\nswing window = {SWING_FRAC[0]:.2f}-{SWING_FRAC[1]:.2f} of crouch->peak, "
          f"from the hand-picked f320-400 on the earlier cut")
