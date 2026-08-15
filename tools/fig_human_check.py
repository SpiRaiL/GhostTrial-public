#!/usr/bin/env python3
"""Figure: does the reconstructed human hold up as a human?

    .venv/bin/python tools/fig_human_check.py "<report dir>/fig_human_check.svg"

Six independent checks, each on its own scale with the plausible band shaded, and
one dot per take. Separate scales rather than one normalised axis because the
question is per-metric — "is this in range" — not "which metric is biggest".
Every dot carries its number, so the pass/warn colour is never the only signal.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import BVHDIR, report  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_human_check.svg"

BG, INK, DIM, GRID = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
GOOD, WARN, BAD, BAND = "#4ec9a3", "#e6b34d", "#e0736b", "#1b2a2a"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

files = sorted(f for f in os.listdir(BVHDIR) if f.endswith(".bvh"))
M = {f[0]: report(os.path.join(BVHDIR, f)) for f in files}   # keyed A / B / C

# (label, unit, accessor, axis lo, axis hi, band lo, band hi, source of the band)
ROWS = [
    ("Stature", "m", lambda d: d["height"], 1.55, 1.95, 1.60, 1.90,
     "adult male range"),
    ("Leg length", "% of stature", lambda d: d["leg_pct"], 40, 58, 48, 52,
     "thigh + shin vs norm"),
    ("Arm length", "% of stature", lambda d: d["arm_pct"], 24, 38, 31, 34,
     "upper arm + forearm vs norm"),
    ("Left/right asymmetry", "%", lambda d: d["asym"] * 100, 0, 8, 0, 4,
     "one identity solves both sides"),
    ("Foot through floor", "mm", lambda d: d["pen"] * 1000, 0, 30, 0, 10,
     "deepest ground penetration"),
    ("Planted-foot slide", "m/s", lambda d: d["slide"], 0, 0.6, 0, 0.20,
     "p90 speed of a grounded foot"),
]

W, RH, TOP = 1060, 66, 96
H = TOP + RH * len(ROWS) + 34
X0, X1 = 300, 700
XR = 754                                     # readout column

p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
     f'height="{H}" font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\','
     f'Roboto,Helvetica,Arial,sans-serif"><rect width="{W}" height="{H}" fill="{BG}"/>',
     f'<text x="24" y="32" fill="{INK}" font-size="15" font-weight="600">'
     f'The reconstructed human measures like a human</text>',
     f'<text x="24" y="52" fill="{DIM}" font-size="12">six checks on the SOMA skeleton '
     f'GEM-X solved, before any robot is involved · shaded band = plausible range · '
     f'one dot per take</text>',
     f'<text x="24" y="72" fill="{DIM}" font-size="11.5">A deep-crouch · B hold · '
     f'C 45° — all three solved independently from separate clips, so agreement '
     f'between them is evidence the solve is stable</text>']

for i, (lab, unit, get, lo, hi, blo, bhi, why) in enumerate(ROWS):
    y = TOP + i * RH + 26

    def sx(v):
        return X0 + (max(lo, min(hi, v)) - lo) / (hi - lo) * (X1 - X0)

    p.append(f'<text x="{X0 - 20}" y="{y - 2}" fill="{INK}" font-size="12.5" '
             f'text-anchor="end" font-weight="600">{lab}</text>')
    p.append(f'<text x="{X0 - 20}" y="{y + 14}" fill="{DIM}" font-size="10.5" '
             f'text-anchor="end">{why}</text>')
    # the band, then the axis line over it
    p.append(f'<rect x="{sx(blo):.1f}" y="{y - 13}" width="{sx(bhi) - sx(blo):.1f}" '
             f'height="26" rx="4" fill="{BAND}"/>')
    p.append(f'<line x1="{X0}" y1="{y}" x2="{X1}" y2="{y}" stroke="{GRID}" '
             f'stroke-width="1"/>')
    for e, v, a in ((X0, lo, "start"), (X1, hi, "end")):
        p.append(f'<text x="{e}" y="{y + 26}" fill="{DIM}" font-size="10" '
                 f'text-anchor="{a}" font-family="{MONO}">{v:g}</text>')
    p.append(f'<text x="{(X0 + X1) / 2}" y="{y + 26}" fill="{DIM}" font-size="10" '
             f'text-anchor="middle">{unit}</text>')

    # dots on the axis; the numbers live in a fixed readout column so they can
    # never collide, however close two takes land
    for k, (take, d) in enumerate(M.items()):
        v = get(d)
        col = GOOD if blo <= v <= bhi else (WARN if v <= bhi * 1.6 else BAD)
        p.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="6.5" fill="{col}" '
                 f'stroke="{BG}" stroke-width="2"/>')
        rx = XR + k * 100
        p.append(f'<text x="{rx}" y="{y + 4}" fill="{DIM}" font-size="11" '
                 f'font-family="{MONO}">{take}</text>')
        p.append(f'<text x="{rx + 16}" y="{y + 4}" fill="{col}" font-size="11.5" '
                 f'font-weight="600" font-family="{MONO}">'
                 f'{v:.{2 if hi <= 10 else 1}f}</text>')

p.append(f'<text x="24" y="{H - 12}" fill="{DIM}" font-size="11">'
         f'measured by tools/check_human_bvh.py — generic BVH parse + forward '
         f'kinematics, no GEM-X code in the loop</text></svg>')

open(OUT, "w").write("\n".join(p))
print(f"wrote {OUT}")
