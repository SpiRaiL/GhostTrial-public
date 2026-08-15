#!/usr/bin/env python3
"""Figures for the capture-review section: angle coverage and hold-at-extension.

    .venv/bin/python tools/make_capture_fig.py "<report dir>"
"""

import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

BG, INK, DIM, LINE = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
ACCENT, GOOD, BAD, WARN = "#59b0ff", "#4ec9a3", "#e0736b", "#e6b34d"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

V = json.load(open(f"{REPO}/data/capture_analysis/verdict.json"))
ANG = {"Angled view": 22, "Side view": 30, "Front view 01": 69, "Front view 02": 74}


def T(x, y, t, fill=INK, size=13, anchor="start", weight="400", mono=False, rot=None):
    fam = f' font-family="{MONO}"' if mono else ""
    r = f' transform="rotate({rot} {x} {y})"' if rot else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam}{r}>{t}</text>')


# ── figure: camera angle coverage ────────────────────────────────────────────
W, H = 1000, 300
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
     f'height="{H}" font-family="{FONT}"><rect width="{W}" height="{H}" fill="{BG}"/>']
s.append(T(24, 30, "Angle coverage — the two extremes, nothing in the middle", INK, 15, weight="600"))
s.append(T(24, 50, "measured from shoulder span during the neutral stance; 0° = camera square to "
                   "the performer, 90° = pure profile", DIM, 12))

L, R2 = 90, 60
pw = W - L - R2
ax_y = 190


def X(a):
    return L + pw * a / 90.0


s.append(f'<line x1="{L}" y1="{ax_y}" x2="{L+pw}" y2="{ax_y}" stroke="{LINE}" stroke-width="2"/>')
for a in (0, 15, 30, 45, 60, 75, 90):
    s.append(f'<line x1="{X(a):.1f}" y1="{ax_y-6}" x2="{X(a):.1f}" y2="{ax_y+6}" stroke="{DIM}" stroke-width="1"/>')
    s.append(T(X(a), ax_y + 24, f"{a}°", DIM, 12, "middle", mono=True))

# the wanted band
s.append(f'<rect x="{X(35):.1f}" y="{ax_y-96}" width="{X(55)-X(35):.1f}" height="96" '
         f'fill="{GOOD}" opacity="0.11"/>')
s.append(T((X(35) + X(55)) / 2, ax_y - 104, "asked for: ~45°, most reps here", GOOD, 12.5, "middle", weight="600"))

for i, (name, a) in enumerate(sorted(ANG.items(), key=lambda kv: kv[1])):
    y = ax_y - 74 + (i % 2) * 26
    col = BAD if (a < 35 or a > 55) else GOOD
    s.append(f'<line x1="{X(a):.1f}" y1="{y+8}" x2="{X(a):.1f}" y2="{ax_y}" stroke="{col}" stroke-width="1.5" opacity="0.6"/>')
    s.append(f'<circle cx="{X(a):.1f}" cy="{ax_y}" r="6" fill="{col}" stroke="{BG}" stroke-width="2"/>')
    n = len(V[name]["takes"])
    s.append(T(X(a), y, f"{name} · {n} takes", col, 12.5, "middle", weight="600"))

s.append(T(24, H - 26, "Two clips sit near front-on and two near profile. Front-on hides depth along "
                       "the throw axis; profile hides the far arm and leg.", DIM, 12))
s.append(T(24, H - 8, "Not fatal — 40 usable takes either way — but a 45° pass is the one that would "
                      "most improve the 3D solve.", WARN, 12.5, weight="600"))
s.append("</svg>")
open(f"{OUT}/fig_capture_angles.svg", "w").write("\n".join(s))
print("wrote", f"{OUT}/fig_capture_angles.svg")

# ── figure: hold at full extension, per take ─────────────────────────────────
rows = []
for clip, d in V.items():
    for i, tk in enumerate(d["takes"], 1):
        rows.append((clip, i, tk["dur"], tk["deepest_crouch"], tk["wrist_above_head"]))

# hold numbers recomputed in capture_verdict are per-clip; use the Angled set measured earlier
HOLD = {  # take index -> seconds, Angled view (the strongest angle)
    1: 0.15, 2: 0.60, 3: 0.60, 4: 0.75, 5: 0.05, 6: 0.40, 7: 0.05, 8: 0.60, 9: 0.05}

W2, H2 = 1000, 300
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}" width="{W2}" '
     f'height="{H2}" font-family="{FONT}"><rect width="{W2}" height="{H2}" fill="{BG}"/>']
s.append(T(24, 30, "The hold at full extension is short, and inconsistent take to take", INK, 15, weight="600"))
s.append(T(24, 50, "seconds the reaching hand stays within 90% of that take's peak extension "
                   "· Angled view, all 9 takes", DIM, 12))
L3, B3 = 70, 90
pw3 = W2 - L3 - 240
ph3 = 150
mx = 1.6


def Y3(v):
    return B3 + ph3 - ph3 * v / mx


for gv in (0, 0.5, 1.0, 1.5):
    s.append(f'<line x1="{L3}" y1="{Y3(gv):.1f}" x2="{L3+pw3}" y2="{Y3(gv):.1f}" stroke="{LINE}" stroke-width="1"/>')
    s.append(T(L3 - 10, Y3(gv) + 4, f"{gv:.1f}", DIM, 11.5, "end", mono=True))
s.append(T(L3 - 10, B3 - 10, "sec", DIM, 11, "end"))

bw = pw3 / len(HOLD)
for i, (k, v) in enumerate(sorted(HOLD.items())):
    x = L3 + bw * i + bw * 0.2
    col = BAD if v < 0.3 else (WARN if v < 0.6 else GOOD)
    s.append(f'<rect x="{x:.1f}" y="{Y3(v):.1f}" width="{bw*0.6:.1f}" height="{B3+ph3-Y3(v):.1f}" rx="4" fill="{col}"/>')
    s.append(T(x + bw * 0.3, B3 + ph3 + 18, f"{k}", DIM, 11.5, "middle", mono=True))
    s.append(T(x + bw * 0.3, Y3(v) - 6, f"{v:.2f}", col, 11, "middle", mono=True, weight="600"))
s.append(T(L3 + pw3 / 2, B3 + ph3 + 38, "take", DIM, 12, "middle"))

ry = Y3(1.5)
s.append(f'<line x1="{L3}" y1="{ry:.1f}" x2="{L3+pw3}" y2="{ry:.1f}" stroke="{ACCENT}" '
         f'stroke-width="2" stroke-dasharray="6 4"/>')
s.append(T(L3 + pw3 + 12, ry + 4, "the reference holds ~1.5 s", ACCENT, 12.5, weight="600"))
s.append(T(L3 + pw3 + 12, ry + 24, "4 of 9 takes hold &lt; 0.2 s", BAD, 12.5, weight="600"))
s.append(T(L3 + pw3 + 12, ry + 44, "best take: 0.75 s", GOOD, 12.5, weight="600"))
s.append(T(24, H2 - 12, "Everything else about the performance is consistent — this is the one "
                        "thing worth a reshoot note.", DIM, 12))
s.append("</svg>")
open(f"{OUT}/fig_capture_hold.svg", "w").write("\n".join(s))
print("wrote", f"{OUT}/fig_capture_hold.svg")
