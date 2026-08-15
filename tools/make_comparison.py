#!/usr/bin/env python3
"""Side-by-side: the original footage vs the commissioned performance.

    .venv/bin/python tools/make_comparison.py "<report dir>"

Uses the SIDE VIEW. In MK1 terms "side" is relative to the direction of the fight —
forward is toward the opponent — and in a bladed fighting stance that is not where
the chest points. The side view is therefore the one that shows the throw travelling
across frame, which is both the game's camera and the best view for the 3D solve.

Frames are matched by PHASE, not by clock — the performances run at different
speeds — and each is cropped to the performer so both rows show the body at a
comparable size. Phase times come from the keypoints, not from eyeballing.

Outputs
  compare_throw.png      10 phases, original over actor
  compare_uppercut.png   10 phases, original over actor
  compare_notes.png      the three differences, zoomed, with the measurements
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
TMP = "/tmp/cmp"
os.makedirs(TMP, exist_ok=True)

KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb",
      "l_wri", "r_wri", "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
I = {n: i for i, n in enumerate(KP)}
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_T, F_L, F_N, F_BIG = (ImageFont.truetype(FB, s) for s in (26, 17, 15, 40))
TH = 380
GOOD, BAD, WARN, INK, DIM = (78, 201, 163), (224, 115, 107), (230, 179, 77), (231, 237, 243), (154, 167, 180)


def person_box(npy, fps, win, srcw, srch, kpw=640, kph=360):
    """Crop sized from the performer's STANDING HEIGHT, not their travel.

    Sizing from the union bbox makes the box as wide as the performer walks, which
    shrinks them to nothing when they cover ground. Height-based sizing keeps the
    body the same size in both rows, which is the whole point of the comparison.
    """
    a = np.load(npy)
    t = a[:, 0] / fps
    s = (t >= win[0]) & (t <= win[1])
    x, y = a[s, 1:18], a[s, 18:35]
    sx, sy = srcw / kpw, srch / kph
    head = np.nanmin(y, axis=1) * sy
    feet = np.nanmax(y, axis=1) * sy
    stand = float(np.nanpercentile(feet - head, 90))
    h = stand * 1.42
    w = h * 0.86
    cx = float(np.nanmedian(np.nanmean(x[:, [I["l_hip"], I["r_hip"]]], axis=1))) * sx
    floor = float(np.nanpercentile(feet, 96))
    y0 = min(max(0, floor + stand * 0.10 - h), srch - h)
    x0 = min(max(0, cx - w / 2), srcw - w)
    return int(x0), int(y0), int(w), int(h)


def grab(src, t, box, out, h=TH):
    x, y, w, hh = box
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", src,
                    "-vframes", "1", "-vf", f"crop={w}:{hh}:{x}:{y},scale=-2:{h}", out],
                   check=True)
    return Image.open(out).convert("RGB")


def strip(title, sub, rows, path, notes=None):
    n = max(len(r[1]) for r in rows)
    tw = max(im.width for _, ims in rows for _, _, im in ims)
    head, cap, lab = 68, 30, 132
    W = lab + tw * n
    H = head + len(rows) * (TH + cap) + (46 if notes else 8)
    sheet = Image.new("RGB", (W, H), (15, 18, 22))
    d = ImageDraw.Draw(sheet)
    d.text((22, 12), title, fill=INK, font=F_T)
    d.text((22, 44), sub, fill=DIM, font=F_L)
    for r, (rlabel, ims) in enumerate(rows):
        y = head + r * (TH + cap)
        d.text((14, y + TH // 2 - 12), rlabel, fill=(255, 220, 90), font=F_L)
        for c, (capn, val, im) in enumerate(ims):
            X = lab + tw * c
            sheet.paste(im, (X, y))
            d.text((X + 8, y + TH + 6), capn, fill=DIM, font=F_N)
            if val:
                txt, col = val
                bb = d.textbbox((0, 0), txt, font=F_N)
                d.rectangle([X + 6, y + 6, X + 14 + bb[2], y + 14 + bb[3]], fill=(0, 0, 0))
                d.text((X + 10, y + 10), txt, fill=col, font=F_N)
    if notes:
        d.text((22, H - 34), notes, fill=WARN, font=F_L)
    sheet.save(path)
    print("wrote", path, sheet.size)


REF_TH = f"{REPO}/raw/reference-clips/ref_throw.mov"
REF_UP = f"{REPO}/raw/reference-clips/ref_uppercut.mov"
ACT = f"{REPO}/raw/capture-01/Side view.mov"
PH = json.load(open("/tmp/side_phase.json"))

box_th = person_box(f"{REPO}/data/reference_analysis/ref_throw_kp.npy", 30, (0, 6), 962, 720)
box_up = person_box(f"{REPO}/data/reference_analysis/ref_uppercut_kp.npy", 30, (0, 6), 960, 720)
box_ac = person_box(f"{REPO}/data/capture_analysis/Side view_kp.npy", 59.94, (PH["s"], PH["e"]),
                    3840, 2160)

# ── the throw, 10 phases ─────────────────────────────────────────────────────
r_th = [0.60, 1.60, 2.10, 2.60, 3.00, 3.30, 3.70, 4.10, 4.60, 5.10]
a0, a1 = PH["s"] + 0.2, PH["throw"]
a_th = [a0, a0 + (a1 - a0) * .45, a0 + (a1 - a0) * .70, a0 + (a1 - a0) * .88, a1 - 0.06,
        a1, a1 + 0.18, a1 + 0.42, a1 + 0.75, a1 + 1.15]
caps = ["settled", "shift", "coil", "coil peak", "fire", "EXTENSION", "hold", "hold", "release", "guard"]
vals_r = [None] * 10
vals_a = [None] * 10
vals_r[3] = ("hand chambered at shoulder", GOOD)
vals_a[3] = ("same — matches", GOOD)
vals_r[5] = ("holds ~1.5 s", GOOD)
vals_a[5] = ("holds 0.40 s", WARN)
rows = [
    ("ORIGINAL", [(c, v, grab(REF_TH, t, box_th, f"{TMP}/tr{i}.png"))
                  for i, (c, v, t) in enumerate(zip(caps, vals_r, r_th))]),
    ("ACTOR", [(c, v, grab(ACT, t, box_ac, f"{TMP}/ta{i}.png"))
               for i, (c, v, t) in enumerate(zip(caps, vals_a, a_th))]),
]
strip("THE THROW — original vs actor, matched by phase",
      "original 2:00–2:06  ·  actor Side view, take 71.3–77.7 s  ·  both cropped to the performer",
      rows, f"{OUT}/compare_throw.png",
      "The throw is close. Stance, chamber, extension line and release all match. The one gap is the "
      "HOLD at full extension - 0.40 s here against ~1.5 s in the original.")

# ── the uppercut, 10 phases ──────────────────────────────────────────────────
r_up = [1.40, 2.20, 2.60, 2.90, 3.10, 3.50, 3.95, 4.40, 5.00, 5.60]
b1, b2 = PH["crouch"], PH["rise"]
b0 = b1 - 1.7          # the actor's sink is fast; sample it from further back
a_up = [b0, b0 + (b1 - b0) * .45, b0 + (b1 - b0) * .70, b0 + (b1 - b0) * .88, b1,
        b1 + (b2 - b1) * .28, b1 + (b2 - b1) * .5, b1 + (b2 - b1) * .72, b2 - 0.10, b2]
caps = ["guard", "sink", "sink", "sink", "DEEPEST", "drive", "drive", "rise", "rise", "OVERHEAD"]
vals_r = [None] * 10
vals_a = [None] * 10
vals_r[4] = ("hips drop to 29% of standing", GOOD)
vals_a[4] = ("hips stay at 53%", BAD)
vals_r[9] = ("+0.15 above head", GOOD)
vals_a[9] = ("+0.20 — matches", GOOD)
rows = [
    ("ORIGINAL", [(c, v, grab(REF_UP, t, box_up, f"{TMP}/ur{i}.png"))
                  for i, (c, v, t) in enumerate(zip(caps, vals_r, r_up))]),
    ("ACTOR", [(c, v, grab(ACT, t, box_ac, f"{TMP}/ua{i}.png"))
               for i, (c, v, t) in enumerate(zip(caps, vals_a, a_up))]),
]
strip("THE UPPERCUT — original vs actor, matched by phase",
      "original 2:53–2:59  ·  actor Side view, same take  ·  the sink is the difference",
      rows, f"{OUT}/compare_uppercut.png",
      "THE one note: the original sits DOWN into the crouch, the actor folds forward at the waist. "
      "Every one of the 38 takes is shallower than the original. Drive and overhead finish match well.")

# ── the three notes, zoomed ──────────────────────────────────────────────────
NOTES = [
    ("1 · Sit DOWN into the crouch — every take", REF_UP, 3.10, ACT, PH["crouch"],
     "hips drop to 29% of standing", "hips stay at 53%",
     "38 of 38 takes shallower",
     "Bend the knees and drop the hips rather than folding at the waist. This is the big one."),
    ("2 · Hold the extension longer", REF_TH, 3.30, ACT, PH["throw"],
     "holds ~1.5 s at full reach", "holds 0.40 s",
     "the shape is right, the dwell is short",
     "Freeze at full extension for a slow count of two before releasing."),
    ("3 · Play it wider, and let it settle", REF_TH, 0.60, ACT, PH["s"] + 0.2,
     "ankles 0.51-0.59 apart", "0.34 - ~40% narrower",
     "21% of frames still, vs 28-33%",
     "A wider base gives somewhere to drop into, and a beat of stillness between moves."),
]
tiles = []
for (title, rsrc, rt, asrc, at, rlab, alab, num, fix) in NOTES:
    rb = box_th if rsrc == REF_TH else box_up
    ri = grab(rsrc, rt, rb, f"{TMP}/n_r_{title[0]}.png", 320)
    ai = grab(asrc, at, box_ac, f"{TMP}/n_a_{title[0]}.png", 320)
    tiles.append((title, ri, ai, rlab, alab, num, fix))

IMH = 320
GAP = 14
imw = max(t[1].width + t[2].width for t in tiles) + GAP
TXT = 620
PW = 24 + imw + 28 + TXT + 24
PH_ = IMH + 74
sheet = Image.new("RGB", (PW, PH_ * len(tiles) + 92), (15, 18, 22))
d = ImageDraw.Draw(sheet)
d.text((24, 16), "The three things worth changing", fill=INK, font=F_T)
d.text((24, 52), "everything else - stance, chamber, extension line, drive, overhead finish, phrasing - already matches",
       fill=DIM, font=F_L)
for k, (title, ri, ai, rlab, alab, num, fix) in enumerate(tiles):
    y = 92 + k * PH_
    sheet.paste(ri, (24, y))
    sheet.paste(ai, (24 + ri.width + GAP, y))
    d.text((30, y + 6), "ORIGINAL", fill=(255, 220, 90), font=F_N)
    d.text((30 + ri.width + GAP, y + 6), "ACTOR", fill=(255, 220, 90), font=F_N)
    for tx, lab, col in ((30, rlab, GOOD), (30 + ri.width + GAP, alab, BAD)):
        bb = d.textbbox((0, 0), lab, font=F_N)
        d.rectangle([tx - 4, y + IMH - 26, tx + bb[2] + 8, y + IMH - 4], fill=(0, 0, 0))
        d.text((tx, y + IMH - 24), lab, fill=col, font=F_N)
    tx = 24 + imw + 28
    d.text((tx, y + 4), title, fill=WARN, font=F_T)
    d.text((tx, y + 44), num, fill=INK, font=F_L)
    # wrap the fix line
    words, line, ly = fix.split(), "", y + 78
    for w in words:
        trial = (line + " " + w).strip()
        if d.textbbox((0, 0), trial, font=F_L)[2] > TXT:
            d.text((tx, ly), line, fill=DIM, font=F_L)
            ly += 26
            line = w
        else:
            line = trial
    d.text((tx, ly), line, fill=DIM, font=F_L)
    if k < len(tiles) - 1:
        d.line([(24, y + PH_ - 26), (PW - 24, y + PH_ - 26)], fill=(42, 51, 61), width=1)
sheet.save(f"{OUT}/compare_notes.png")
print("wrote", f"{OUT}/compare_notes.png", sheet.size)
