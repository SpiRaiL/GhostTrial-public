#!/usr/bin/env python3
"""Turn hand-picked frame pairs into the comparison sequence.

    .venv/bin/python tools/make_sequence.py "<report dir>"

Reads data/sync_pairs.json — pairs chosen by hand in tools/sync_ui.py, because
automatic phase matching kept picking the wrong beat. Emits:

  sequence.png        one row per pair: reference | actor | note.  The sendable one.
  sequence_strip.png  compact two-row filmstrip, for seeing the motion progress.

Frames come from the ORIGINAL media, not the 720p proxies, so the output is full
quality; the proxies only ever existed so the browser could scrub.
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = "/tmp/seq"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
F_T = ImageFont.truetype(FB, 30)
F_H = ImageFont.truetype(FB, 20)
F_N = ImageFont.truetype(FB, 17)
F_B = ImageFont.truetype(FR, 18)
F_S = ImageFont.truetype(FB, 15)
INK, DIM, WARN, GOOD, BAD, ACC = ((231, 237, 243), (154, 167, 180), (230, 179, 77),
                                  (78, 201, 163), (224, 115, 107), (89, 176, 255))
BG = (15, 18, 22)

# proxy name -> (original media, keypoints for framing, its fps)
SRC = {
    "ref_throw.mp4": (f"{REPO}/raw/reference-clips/ref_throw.mov",
                      f"{REPO}/data/reference_analysis/ref_throw_kp.npy", 30.0),
    "ref_uppercut.mp4": (f"{REPO}/raw/reference-clips/ref_uppercut.mov",
                         f"{REPO}/data/reference_analysis/ref_uppercut_kp.npy", 30.0),
    "actor_side_view.mp4": (f"{REPO}/raw/capture-01/Side view.mov",
                            f"{REPO}/data/capture_analysis/Side view_kp.npy", 59.94),
    "actor_angled_view.mp4": (f"{REPO}/raw/capture-01/Angled view.mov",
                              f"{REPO}/data/capture_analysis/Angled view_kp.npy", 59.94),
    "actor_front_view_01.mp4": (f"{REPO}/raw/capture-01/Front view 01.mov",
                                f"{REPO}/data/capture_analysis/Front view 01_kp.npy", 59.94),
    "actor_front_view_02.mp4": (f"{REPO}/raw/capture-01/Front view 02.mov",
                                f"{REPO}/data/capture_analysis/Front view 02_kp.npy", 59.94),
}
KPW, KPH = 640, 360


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout.strip())


def dims(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip().strip(",")
    w, h = r.split(",")[:2]
    return int(w), int(h)


def box_for(npy, fps, t0, t1, srcw, srch):
    """Crop sized from standing height so both rows show the body the same size."""
    a = np.load(npy)
    t = a[:, 0] / fps
    s = (t >= t0 - 0.6) & (t <= t1 + 0.6)
    if s.sum() < 4:
        s = np.ones(len(t), bool)
    x, y = a[s, 1:18], a[s, 18:35]
    sx, sy = srcw / KPW, srch / KPH
    head = np.nanmin(y, axis=1) * sy
    feet = np.nanmax(y, axis=1) * sy
    stand = float(np.nanpercentile(feet - head, 90))
    h = min(stand * 1.45, srch)
    w = min(h * 0.88, srcw)
    cx = float(np.nanmedian(np.nanmean(x[:, [11, 12]], axis=1))) * sx
    floor = float(np.nanpercentile(feet, 96))
    y0 = min(max(0, floor + stand * 0.10 - h), srch - h)
    x0 = min(max(0, cx - w / 2), srcw - w)
    return int(x0), int(y0), int(w), int(h)


def grab(src, t, box, out, th, dur=None, fps=30.0):
    # a chosen frame can sit exactly on (or a hair past) the last frame; ffmpeg then
    # writes nothing at all. Clamp to just inside the clip.
    if dur is not None:
        t = min(t, max(0.0, dur - 1.5 / fps))
    x, y, w, h = box
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", src,
                    "-vframes", "1", "-vf", f"crop={w}:{h}:{x}:{y},scale=-2:{th}", out],
                   check=True)
    return Image.open(out).convert("RGB")


pairs = json.load(open(f"{REPO}/data/sync_pairs.json"))
print(f"{len(pairs)} pairs")

# one crop box per clip, spanning every frame that clip is used for
spans = {}
for p in pairs:
    for side in ("ref", "act"):
        c, t = p[f"{side}_clip"], p[f"{side}_t"]
        lo, hi = spans.get(c, (t, t))
        spans[c] = (min(lo, t), max(hi, t))
boxes = {}
for c, (t0, t1) in spans.items():
    media, npy, fps = SRC[c]
    W, H = dims(media)
    boxes[c] = (media, box_for(npy, fps, t0, t1, W, H), duration(media), fps)

# ── sequence.png — one row per pair, with the note ───────────────────────────
TH = 300
rows = []
for i, p in enumerate(pairs, 1):
    rm, rb, rd, rf = boxes[p["ref_clip"]]
    am, ab, ad, af = boxes[p["act_clip"]]
    ri = grab(rm, p["ref_t"], rb, f"{TMP}/r{i}.png", TH, rd, rf)
    ai = grab(am, p["act_t"], ab, f"{TMP}/a{i}.png", TH, ad, af)
    rows.append((i, p, ri, ai))
    print(f"  {i:2d} {p['comment'][:60]}")

imw = max(r[2].width + r[3].width for r in rows) + 12
TXT = 700
PW = 26 + imw + 30 + TXT + 26
RH = TH + 34
sheet = Image.new("RGB", (PW, 118 + RH * len(rows)), BG)
d = ImageDraw.Draw(sheet)
d.text((26, 22), "Reference vs actor — hand-aligned beats", fill=INK, font=F_T)
d.text((26, 62), f"{len(rows)} pairs chosen frame by frame in tools/sync_ui.py  ·  "
                 f"left: original MK footage  ·  right: actor, side view", fill=DIM, font=F_B)
d.text((26, 86), "the side view is the fight-relative camera — forward is toward the opponent, "
                 "which in a bladed stance is not where the chest points", fill=DIM, font=F_S)

for k, (i, p, ri, ai) in enumerate(rows):
    y = 118 + k * RH
    sheet.paste(ri, (26, y))
    sheet.paste(ai, (26 + ri.width + 12, y))
    d.rectangle([26, y, 26 + ri.width - 1, y + TH - 1], outline=(42, 51, 61))
    d.rectangle([26 + ri.width + 12, y, 26 + ri.width + 11 + ai.width, y + TH - 1],
                outline=(42, 51, 61))
    tx = 26 + imw + 30
    d.text((tx, y + 2), f"{i}", fill=ACC, font=F_T)
    d.text((tx + 44, y + 10), f"ref {p['ref_clip'][:-4]}  f{p['ref_frame']}", fill=DIM, font=F_S)
    d.text((tx + 44, y + 30), f"actor f{p['act_frame']}  ·  {p['act_t']:.2f}s", fill=DIM, font=F_S)
    # wrap the comment
    words, line, ly = (p.get("comment") or "").split(), "", y + 62
    for w in words:
        trial = (line + " " + w).strip()
        if d.textbbox((0, 0), trial, font=F_H)[2] > TXT - 50:
            d.text((tx + 44, ly), line, fill=INK, font=F_H)
            ly += 28
            line = w
        else:
            line = trial
    if line:
        d.text((tx + 44, ly), line, fill=INK, font=F_H)
    if k < len(rows) - 1:
        d.line([(26, y + RH - 16), (PW - 26, y + RH - 16)], fill=(35, 42, 51))
sheet.save(f"{OUT}/sequence.png")
print("wrote", f"{OUT}/sequence.png", sheet.size)

# ── sequence_strip.png — compact filmstrip ───────────────────────────────────
STH = 260
tiles = []
for i, p, ri, ai in rows:
    tiles.append((i, ri.resize((int(ri.width * STH / TH), STH)),
                  ai.resize((int(ai.width * STH / TH), STH))))
tw = max(max(t[1].width, t[2].width) for t in tiles)
lab = 96
W2 = lab + tw * len(tiles)
H2 = 92 + 2 * (STH + 26)
strip = Image.new("RGB", (W2, H2), BG)
d = ImageDraw.Draw(strip)
d.text((22, 18), "The phrase, beat by beat — original above, actor below", fill=INK, font=F_T)
d.text((22, 58), f"{len(tiles)} hand-aligned pairs, left to right", fill=DIM, font=F_B)
for r, which in enumerate((1, 2)):
    y = 92 + r * (STH + 26)
    d.text((16, y + STH // 2 - 10), ("ORIGINAL", "ACTOR")[r], fill=(255, 220, 90), font=F_N)
    for c, t in enumerate(tiles):
        strip.paste(t[which], (lab + tw * c, y))
        if r == 0:
            d.text((lab + tw * c + 6, y + 4), str(t[0]), fill=ACC, font=F_N)
strip.save(f"{OUT}/sequence_strip.png")
print("wrote", f"{OUT}/sequence_strip.png", strip.size)
