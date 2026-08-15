#!/usr/bin/env python3
"""Re-cut the takes to the WHOLE phrase, wind-up included.

    .venv/bin/python tools/cut_full_phrase.py [--write]

The take windows in data/capture02_analysis/takes_ranked.json were chosen by a
scorer that maximises crouch depth, reach, overhead clearance and hold — all of
which peak during the uppercut half. So each window opens at the instant the
throwing arm reaches full extension and drops everything before it: the settled
stance, the wind-up, and the throw itself. The reconstruction that came out of
those clips therefore starts with the spear already thrown.

This finds the real phrase boundaries instead. The scorer's start time is a
reliable anchor — it lands on the throw extension — so from there we walk
backwards to the last sustained stillness (the idle stance before the wind-up)
and forwards from the scorer's end to the next one (the reset after the
uppercut), and cut on those.
"""

import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(REPO, "data", "capture02_analysis")
RAW = os.path.join(REPO, "raw", "capture-02")
DST = os.path.join(REPO, "data", "gemx_input")
KPW, KPH = 640, 360

# the three takes already chosen, by (name, source clip, scorer window)
TAKES = [
    ("A_side_deepcrouch", "side_revision_01", 49.80, 58.40, 60.0),
    ("B_side_hold",       "side_revision_01",  5.10, 13.15, 60.0),
    ("C_angle45_besthold", "angle_revision_02", 18.97, 27.68, 60000 / 1001),
]
STILL = 0.35      # fraction of median motion energy that counts as "settled"
MIN_STILL = 0.45  # a real idle lasts this long; shorter pauses are beats in the move
LEAD = 0.70       # seconds of that idle kept before the wind-up starts
SEARCH = 6.0      # how far either side of the scorer window to look for the idle


def energy(clip, fps):
    """Smoothed whole-body motion energy, and the frame times it is sampled at."""
    a = np.load(os.path.join(ANA, f"{clip}_kp.npy"))
    t = a[:, 0] / fps
    xy = np.stack([a[:, 1:18], a[:, 18:35]], axis=-1)          # (N, 17, 2)
    xy[:, :, 0] *= 1.0 / KPW
    xy[:, :, 1] *= 1.0 / KPH
    e = np.nanmean(np.linalg.norm(np.diff(xy, axis=0), axis=2), axis=1)
    e = np.convolve(np.nan_to_num(e), np.ones(5) / 5, mode="same")
    return t[:-1], e


def still_runs(t, e, thresh):
    """[(start, end)] of every sustained settled stretch."""
    m = np.concatenate([[False], e < thresh, [False]])
    edges = np.diff(m.astype(int))
    runs = []
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if b - 1 >= a and t[b - 1] - t[a] >= MIN_STILL:
            runs.append((t[a], t[b - 1]))
    return runs


def bound(runs, anchor, before):
    """The idle to cut on: the LAST one ending before the anchor (or first after).

    Walking backwards to the first sub-threshold frame instead lands inside the
    coil — the move has a held beat there that reads as stillness but is part of
    the phrase. Requiring a sustained run steps over it.
    """
    if before:
        cand = [r for r in runs if r[1] <= anchor and r[1] >= anchor - SEARCH]
        if not cand:
            return max(0.0, anchor - 3.0)
        s, e = cand[-1]
        return max(s, e - LEAD)
    cand = [r for r in runs if r[0] >= anchor and r[0] <= anchor + SEARCH]
    if not cand:
        return anchor + 1.0
    s, e = cand[0]
    return min(e, s + LEAD)


def main(write):
    os.makedirs(DST, exist_ok=True)
    for name, clip, s0, s1, fps in TAKES:
        t, e = energy(clip, fps)
        thresh = float(np.median(e)) * STILL
        runs = still_runs(t, e, thresh)
        start = max(0.0, bound(runs, s0, before=True))
        end = bound(runs, s1, before=False)
        src = os.path.join(RAW, f"{clip}.mov")
        print(f"{name:22s} {clip}  scorer {s0:6.2f}-{s1:6.2f} ({s1 - s0:4.1f}s)  ->  "
              f"full {start:6.2f}-{end:6.2f} ({end - start:4.1f}s)   "
              f"+{s0 - start:.1f}s of wind-up recovered")
        if write:
            out = os.path.join(DST, f"{name}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
                 "-i", src, "-t", f"{end - start:.3f}",
                 "-vf", "scale=1920:-2", "-r", "60",
                 "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-an", out],
                check=True)
            print(f"{'':22s} wrote {out}")


if __name__ == "__main__":
    main("--write" in sys.argv)
