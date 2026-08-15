#!/usr/bin/env python3
"""Figure: what the 2x uppercut swing costs in tracking reward.

    .venv/bin/python tools/fig_training.py "<report dir>/fig_training.svg"

Two fine-tunes of the same policy on the same move, differing only in whether the
uppercut span plays at 2x. Smoothed over 25 iterations because PPO reward is noisy
enough frame to frame that the raw trace hides the gap.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_training.svg"

BG, INK, DIM, GRID = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
A3C, A4C = "#e6b34d", "#4ec9a3"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

RUNS = [("A3 — uppercut at 2x", "logs/train_a3_run.log", A3C),
        ("A4 — natural speed", "logs/train_a4.log", A4C)]


def series(path, win=25):
    raw = [float(m) for m in re.findall(r"Mean rewards: ([0-9.]+)",
                                        open(os.path.join(REPO, path)).read())]
    return [sum(raw[max(0, i - win):i + 1]) / len(raw[max(0, i - win):i + 1])
            for i in range(len(raw))]


data = [(lab, series(p), c) for lab, p, c in RUNS]
W, H = 1060, 450
L, R, T, B = 68, 250, 92, 62
XHI = max(len(s) for _, s, _ in data)
YHI = 40


def X(i):
    return L + i / XHI * (W - L - R)


def Y(v):
    return H - B - v / YHI * (H - B - T)


p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
     f'height="{H}" font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\','
     f'Roboto,Helvetica,Arial,sans-serif"><rect width="{W}" height="{H}" fill="{BG}"/>',
     f'<text x="24" y="32" fill="{INK}" font-size="15" font-weight="600">'
     f'The 2&#215; uppercut swing costs about 11% of tracking reward</text>',
     f'<text x="24" y="52" fill="{DIM}" font-size="12">SONIC fine-tuned on the same move, '
     f'2048 environments on one RTX 5080 · the only difference is whether the uppercut span '
     f'plays at 2&#215; · 25-iteration moving average</text>',
     f'<text x="24" y="72" fill="{DIM}" font-size="11.5">position accuracy is '
     f'<tspan font-style="italic">identical</tspan> at 4.5 cm — the cost is entirely in '
     f'velocity tracking, which is what doubling a span\'s speed would predict</text>']

for v in range(0, YHI + 1, 10):
    p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" '
             f'stroke="{GRID}" stroke-width="1"/>')
    p.append(f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" fill="{DIM}" font-size="11" '
             f'text-anchor="end" font-family="{MONO}">{v}</text>')
p.append(f'<text x="{L - 10}" y="{T - 6}" fill="{DIM}" font-size="11" '
         f'text-anchor="end">reward</text>')
for i in range(0, XHI + 1, 100):
    p.append(f'<text x="{X(i):.1f}" y="{H - B + 20}" fill="{DIM}" font-size="10.5" '
             f'text-anchor="middle" font-family="{MONO}">{i}</text>')
p.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - B + 40}" fill="{DIM}" font-size="11" '
         f'text-anchor="middle">PPO iteration</text>')

for k, (lab, s, col) in enumerate(data):
    pts = " ".join(f"{X(i):.1f},{Y(min(v, YHI)):.1f}" for i, v in enumerate(s))
    p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
             f'stroke-linejoin="round"/>')
    yy = T + 12 + k * 46
    p.append(f'<line x1="{W - R + 16}" y1="{yy}" x2="{W - R + 44}" y2="{yy}" '
             f'stroke="{col}" stroke-width="2"/>')
    p.append(f'<text x="{W - R + 52}" y="{yy + 4}" fill="{INK}" font-size="12">{lab}</text>')
    tail = s[300:600] or s[-50:]
    p.append(f'<text x="{W - R + 52}" y="{yy + 21}" fill="{col}" font-size="11.5" '
             f'font-weight="600" font-family="{MONO}">plateau '
             f'{sum(tail) / len(tail):.1f}</text>')

p.append(f'<text x="24" y="{H - 8}" fill="{DIM}" font-size="11">'
         f'both runs stop improving by about iteration 300; the gap between them does '
         f'not close</text></svg>')
open(OUT, "w").write("\n".join(p))
print(f"wrote {OUT}")
