#!/usr/bin/env python3
"""Turn the saved keypoints into a verdict against the brief.

    .venv/bin/python tools/capture_verdict.py data/capture_analysis

Reads the *_kp.npy written by pose_check.py. Everything here is measured, so the
report comments and the reply to the performer can quote numbers rather than
impressions.
"""

import glob
import json
import os
import sys

import numpy as np

KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb",
      "r_elb", "l_wri", "r_wri", "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
I = {n: i for i, n in enumerate(KP)}
W, H = 640, 360
STRIDE, FPS = 3, 59.94
DT = STRIDE / FPS

D = sys.argv[1] if len(sys.argv) > 1 else "data/capture_analysis"
out = {}

for f in sorted(glob.glob(os.path.join(D, "*_kp.npy"))):
    name = os.path.basename(f).replace("_kp.npy", "")
    a = np.load(f)
    x, y = a[:, 1:18], a[:, 18:35]

    def P(n):
        return np.stack([x[:, I[n]], y[:, I[n]]], axis=1)

    head = y[:, I["nose"]]
    ank_lo = np.nanmax(np.stack([y[:, I["l_ank"]], y[:, I["r_ank"]]]), axis=0)
    hip = np.nanmean(np.stack([y[:, I["l_hip"]], y[:, I["r_hip"]]]), axis=0)
    body = ank_lo - head
    scale = float(np.nanmedian(body))

    # orientation: shoulder separation in x, relative to body height.
    # front-on -> wide; true profile -> near zero.
    sho_w = np.abs(x[:, I["l_sho"]] - x[:, I["r_sho"]]) / body
    orient = float(np.nanmedian(sho_w))

    # highest wrist above the head (positive = above), in body heights
    wri_top = np.nanmin(np.stack([y[:, I["l_wri"]], y[:, I["r_wri"]]]), axis=0)
    above = (head - wri_top) / body
    # forward reach: wrist to shoulder horizontal, body heights
    reach = np.nanmax(np.stack([
        np.abs(x[:, I["r_wri"]] - x[:, I["r_sho"]]),
        np.abs(x[:, I["l_wri"]] - x[:, I["l_sho"]])]), axis=0) / body

    hip_above_floor = ank_lo - hip
    stand = float(np.nanpercentile(hip_above_floor, 90))
    crouch = hip_above_floor / stand           # 1.0 = standing, lower = deeper

    # wrist speed, body heights per second
    wr = P("r_wri")
    v = np.r_[0, np.linalg.norm(np.diff(wr, axis=0), axis=1)] / scale / DT

    # segment takes on sustained wrist motion
    thr = np.nanpercentile(v, 70)
    act = np.convolve(np.nan_to_num(v > thr).astype(float), np.ones(9) / 9, "same") > 0.25
    takes, s, gap = [], None, 0
    for i, aa in enumerate(act):
        if aa:
            s = i if s is None else s
            gap = 0
        elif s is not None:
            gap += 1
            if gap * DT > 0.7:
                if (i - gap - s) * DT > 1.5:
                    takes.append((s, i - gap))
                s = None
                gap = 0
    if s is not None and (len(act) - s) * DT > 1.5:
        takes.append((s, len(act) - 1))

    per = []
    for (s0, s1) in takes:
        sl = slice(s0, s1 + 1)
        per.append(dict(
            t=[round(s0 * DT, 2), round(s1 * DT, 2)],
            dur=round((s1 - s0) * DT, 2),
            deepest_crouch=round(float(np.nanmin(crouch[sl])), 3),
            max_reach=round(float(np.nanmax(reach[sl])), 3),
            wrist_above_head=round(float(np.nanmax(above[sl])), 3),
            peak_wrist_speed=round(float(np.nanmax(v[sl])), 2),
        ))

    out[name] = dict(
        orientation_shoulder_width=round(orient, 3),
        view=("front-on" if orient > 0.20 else "profile" if orient < 0.10 else "three-quarter"),
        subject_height_pct=[round(float(np.nanpercentile(body / H * 100, p)), 1) for p in (5, 50, 95)],
        ankle_margin_px=round(float(H - np.nanmax(ank_lo)), 1),
        head_margin_px=round(float(np.nanmin(head)), 1),
        frames_body_clipped=int(np.nansum((ank_lo > H - 4) | (head < 4))),
        deepest_crouch_overall=round(float(np.nanmin(crouch)), 3),
        max_reach_overall=round(float(np.nanmax(reach)), 3),
        max_wrist_above_head=round(float(np.nanmax(above)), 3),
        peak_wrist_speed=round(float(np.nanmax(v)), 2),
        n_takes=len(takes),
        takes=per,
    )

for k, v in out.items():
    print(f"\n=== {k}")
    print(f"  view: {v['view']} (shoulder width {v['orientation_shoulder_width']} body-heights)")
    print(f"  subject height %frame p5/p50/p95: {v['subject_height_pct']}")
    print(f"  margins: {v['ankle_margin_px']}px below feet, {v['head_margin_px']}px above head, "
          f"clipped frames {v['frames_body_clipped']}")
    print(f"  deepest crouch {v['deepest_crouch_overall']} of standing hip height")
    print(f"  max reach {v['max_reach_overall']} body-heights | wrist above head {v['max_wrist_above_head']}")
    print(f"  peak wrist speed {v['peak_wrist_speed']} body-heights/s")
    print(f"  {v['n_takes']} takes:")
    for t in v["takes"]:
        print(f"     {t['t'][0]:6.2f}-{t['t'][1]:6.2f}s ({t['dur']:4.1f}s) crouch {t['deepest_crouch']:.2f} "
              f"reach {t['max_reach']:.2f} above-head {t['wrist_above_head']:+.2f} peak {t['peak_wrist_speed']:.1f}")

with open(os.path.join(D, "verdict.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote {D}/verdict.json")
