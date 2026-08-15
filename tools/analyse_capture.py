#!/usr/bin/env python3
"""Analyse delivered capture footage against the brief, per frame.

    .venv/bin/python tools/analyse_capture.py raw/capture-01 --out data/capture_analysis

For every file: pipe frames out of ffmpeg at reduced resolution, build a median
background, then per frame measure

  * the performer's bounding box  -> framing, and whether feet/head leave frame
  * motion energy                 -> segments the file into takes
  * blur (variance of Laplacian)  -> catches motion smear on the fast accents

Writes one CSV per clip plus a summary. Nothing here is subjective; the report
comments are written from these numbers.
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

W, H = 480, 270  # analysis resolution; the source is 4K


def frames(path, w=W, h=H):
    """Yield downscaled BGR frames without decoding 4K into memory."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-vf", f"scale={w}:{h}", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.stdout.close()
    p.wait()


def analyse(path):
    # pass 1 — median background from a sparse sample
    sample = []
    for i, f in enumerate(frames(path)):
        if i % 37 == 0:
            sample.append(f)
        if len(sample) >= 90:
            break
    bg = np.median(np.stack(sample), axis=0).astype(np.uint8)
    bgg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)

    rows, prev = [], None
    for i, f in enumerate(frames(path)):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.abs(g - bgg).astype(np.uint8)
        _, m = cv2.threshold(diff, 26, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ys, xs = np.nonzero(m)
        if len(ys) > 150:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            area = len(ys)
        else:
            x0 = x1 = y0 = y1 = -1
            area = 0
        motion = 0.0 if prev is None else float(np.abs(g - prev).mean())
        prev = g
        blur = float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                   cv2.CV_64F).var())
        rows.append((i, x0, y0, x1, y1, area, motion, blur))
    return np.array(rows, dtype=np.float64)


def segment(motion, fps, quiet_frac=0.28, min_gap_s=0.6, min_take_s=2.0):
    """Split into takes on sustained quiet stretches."""
    thr = motion.max() * quiet_frac
    active = motion > thr
    k = max(3, int(fps * 0.25))
    active = np.convolve(active.astype(float), np.ones(k) / k, "same") > 0.25
    takes, start = [], None
    gap = 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > fps * min_gap_s:
                if (i - gap - start) / fps >= min_take_s:
                    takes.append((start, i - gap))
                start = None
                gap = 0
    if start is not None and (len(active) - start) / fps >= min_take_s:
        takes.append((start, len(active) - 1))
    return takes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--out", default="data/capture_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for fn in sorted(os.listdir(args.indir)):
        if not fn.lower().endswith((".mov", ".mp4")):
            continue
        path = os.path.join(args.indir, fn)
        # ffprobe csv=p=0 emits a trailing comma, so parse the fraction by hand
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", path],
            capture_output=True, text=True).stdout.strip().strip(",")
        num, _, den = raw.partition("/")
        fps = float(num) / float(den or 1)
        print(f"\n=== {fn}  ({fps:.2f} fps)", flush=True)
        r = analyse(path)
        np.savetxt(os.path.join(args.out, fn.rsplit(".", 1)[0] + ".csv"), r,
                   delimiter=",",
                   header="frame,x0,y0,x1,y1,area,motion,blur", comments="")
        ok = r[:, 5] > 0
        takes = segment(r[:, 6], fps)
        d = dict(
            fps=fps, n=len(r), duration=len(r) / fps,
            takes=[(round(a / fps, 2), round(b / fps, 2)) for a, b in takes],
            subject_h_pct=[round(float(np.percentile((r[ok, 4] - r[ok, 2]) / H * 100, p)), 1)
                           for p in (5, 50, 95)],
            top_min=int(r[ok, 2].min()), bottom_max=int(r[ok, 4].max()),
            left_min=int(r[ok, 1].min()), right_max=int(r[ok, 3].max()),
            blur=[round(float(np.percentile(r[:, 7], p)), 1) for p in (5, 50, 95)],
        )
        summary[fn] = d
        print(f"  takes: {len(takes)}  {d['takes']}")
        print(f"  subject height %frame (p5/p50/p95): {d['subject_h_pct']}")
        print(f"  bbox extremes: top {d['top_min']} bottom {d['bottom_max']} "
              f"(frame is 0..{H-1});  left {d['left_min']} right {d['right_max']} (0..{W-1})")
        print(f"  sharpness (lap var p5/p50/p95): {d['blur']}")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
