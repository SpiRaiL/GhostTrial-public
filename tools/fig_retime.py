#!/usr/bin/env python3
"""Figure: what the retime changed, and what it left alone.

    .venv/bin/python tools/fig_retime.py "<report dir>/fig_retime.svg"

Punching-hand speed against time, original and edited on the same axes. Both are
aligned at the deepest crouch rather than at frame zero — the edit trims the head
off the clip, so frame numbers no longer correspond, but the crouch is the same
moment in both. Aligning there is what makes "before the swing these are identical"
visible instead of asserted.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_retime.svg"
TAKE = "A_side_deepcrouch_human.bvh"

BG, INK, DIM, GRID = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
ACC, OLD, BAND = "#59b0ff", "#7f8b98", "#1b2530"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def curve(path):
    names, off, par, ch, data, T, fps = parse_bvh(path)
    idx = {n: i for i, n in enumerate(names)}
    pos = fk(names, off, par, ch, data)
    feet = [idx["LeftToeBase"], idx["RightToeBase"]]
    floor = float(np.percentile(pos[:, feet, 1].min(axis=1), 2))
    lift = pos[:, idx["Hips"], 1] - floor
    crouch = int(np.argmin(lift))
    hand = pos[:, idx["RightHand"], 1] - floor
    peak = crouch + int(np.argmax(hand[crouch:]))
    v = np.linalg.norm(np.diff(pos[:, idx["RightHand"]], axis=0), axis=1) * fps
    t = (np.arange(len(v)) - crouch) / fps
    return t, v, (peak - crouch) / fps


t0, v0, _ = curve(os.path.join(REPO, "data", "human_bvh", TAKE))
t1, v1, span = curve(os.path.join(REPO, "data", "human_bvh_edited", TAKE))
# the swing, in the EDITED clip's crouch-relative time — same fractions the retime
# used, halved in duration because that span now plays at 2x
SW0 = 0.217 * span * 2
SW1 = SW0 + (0.797 - 0.217) * span

W, H = 1060, 400
L, R, TOPY, BOT = 66, 250, 96, 60
# span the whole phrase, not just the uppercut — the claim is that everything
# outside the swing window is untouched, and that includes the spear throw
XLO, XHI, YHI = -8.2, 4.8, 15.0


def X(t):
    return L + (t - XLO) / (XHI - XLO) * (W - L - R)


def Y(v):
    return H - BOT - v / YHI * (H - BOT - TOPY)


def poly(t, v, col, wdt):
    m = (t >= XLO) & (t <= XHI)
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t[m], v[m]))
    return (f'<polyline points="{pts}" fill="none" stroke="{col}" '
            f'stroke-width="{wdt}" stroke-linejoin="round"/>')


p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
     f'height="{H}" font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\','
     f'Roboto,Helvetica,Arial,sans-serif"><rect width="{W}" height="{H}" fill="{BG}"/>',
     f'<text x="24" y="32" fill="{INK}" font-size="15" font-weight="600">'
     f'The swing is twice as fast; nothing before it moved</text>',
     f'<text x="24" y="52" fill="{DIM}" font-size="12">punching-hand speed, take A, '
     f'across the whole phrase · outside the shaded span the two curves are identical, '
     f'which is why only one line is visible there</text>',
     f'<text x="24" y="72" fill="{DIM}" font-size="11.5">every pose is preserved '
     f'exactly — this is a pure time warp, so only <tspan font-style="italic">when</tspan> '
     f'each pose happens changed</text>']

# the retimed swing span, in crouch-relative time
p.append(f'<rect x="{X(SW0):.1f}" y="{TOPY}" width="{X(SW1) - X(SW0):.1f}" '
         f'height="{H - BOT - TOPY}" fill="{BAND}"/>')
p.append(f'<text x="{X((SW0 + SW1) / 2):.1f}" y="{TOPY - 8}" fill="{DIM}" font-size="11" '
         f'text-anchor="middle">the swing, at 2&#215;</text>')
for lab, tt in (("wind-up + spear throw", -6.1), ("crouch", 0.0)):
    p.append(f'<line x1="{X(tt):.1f}" y1="{TOPY}" x2="{X(tt):.1f}" y2="{H - BOT}" '
             f'stroke="{GRID}" stroke-width="1" stroke-dasharray="3 3"/>')
    p.append(f'<text x="{X(tt):.1f}" y="{TOPY - 8}" fill="{DIM}" font-size="11" '
             f'text-anchor="middle">{lab}</text>')

for v in range(0, int(YHI) + 1, 5):
    p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" '
             f'stroke="{GRID}" stroke-width="1"/>')
    p.append(f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" fill="{DIM}" font-size="11" '
             f'text-anchor="end" font-family="{MONO}">{v}</text>')
p.append(f'<text x="{L - 10}" y="{TOPY - 10}" fill="{DIM}" font-size="11" '
         f'text-anchor="end">m/s</text>')
for s in np.arange(-8.0, 4.9, 1.0):
    p.append(f'<line x1="{X(s):.1f}" y1="{H - BOT}" x2="{X(s):.1f}" y2="{H - BOT + 5}" '
             f'stroke="{GRID}" stroke-width="1"/>')
    p.append(f'<text x="{X(s):.1f}" y="{H - BOT + 20}" fill="{DIM}" font-size="10.5" '
             f'text-anchor="middle" font-family="{MONO}">{s:+.1f}</text>')
p.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 16}" fill="{DIM}" font-size="11" '
         f'text-anchor="middle">seconds from the deepest crouch</text>')

p.append(poly(t0, v0, OLD, 2))
p.append(poly(t1, v1, ACC, 2))

for lab, t, v, col, dy in (("original", t0, v0, OLD, 0), ("edited", t1, v1, ACC, 48)):
    yy = TOPY + 10 + dy
    p.append(f'<line x1="{W - R + 16}" y1="{yy}" x2="{W - R + 44}" y2="{yy}" '
             f'stroke="{col}" stroke-width="2"/>')
    p.append(f'<text x="{W - R + 52}" y="{yy + 4}" fill="{INK}" font-size="12">{lab}</text>')
    p.append(f'<text x="{W - R + 52}" y="{yy + 21}" fill="{col}" font-size="11.5" '
             f'font-weight="600" font-family="{MONO}">peak {v.max():.2f} m/s</text>')

p.append(f'<text x="{W - R + 16}" y="{H - BOT - 4}" fill="{DIM}" font-size="11">'
         f'{v1.max() / v0.max():.2f}&#215; at the peak —</text>')
p.append(f'<text x="{W - R + 16}" y="{H - BOT + 12}" fill="{DIM}" font-size="11">'
         f'under 2&#215; because</text>')
p.append(f'<text x="{W - R + 16}" y="{H - BOT + 28}" fill="{DIM}" font-size="11">'
         f'slerp resampling rounds</text>')
p.append(f'<text x="{W - R + 16}" y="{H - BOT + 44}" fill="{DIM}" font-size="11">'
         f'the single-frame spike</text>')
p.append("</svg>")

open(OUT, "w").write("\n".join(p))
print(f"wrote {OUT}   original peak {v0.max():.2f} m/s, edited peak {v1.max():.2f} m/s")
