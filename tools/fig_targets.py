#!/usr/bin/env python3
"""Balance margin over time for each candidate target — the fig_static chart, per option.

    MUJOCO_GL=egl .venv/bin/python tools/fig_targets.py

Signed distance from the centre of mass to the edge of the foot support polygon,
every frame. Above the line the pose can be held still; below it the robot is
relying on momentum it does not have, because these clips are slow enough to be
treated as a sequence of static poses.

Plotted against the two clips whose trained reward is known — A9 at 38 and the
BONES-SEED walk at 65.6 — so a candidate can be read as "closer to the thing that
works" rather than judged in isolation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import author_targets as at  # noqa: E402

CLIPS = [
    ("A9 — source (reward 38)", "data/gemx_g1_retimed/A9_smooth.csv", "#e0644a"),
    ("T8 — feet planted, throw arm levelled", "data/gemx_g1_retimed/T8_final.csv", "#c58cf5"),
    ("T10 — both arms level, upright finish", "data/gemx_g1_retimed/T10_upright.csv", "#4ec9a5"),
    ("BONES-SEED walk (reward 65.6)", "data/csv_frozen/ctrl2/CTRL2_walk.csv", "#9aa7b4"),
]
W, H, PAD = 1180, 560, 62


def main():
    rows = [(lab, at.report(p, lab), col) for lab, p, col in CLIPS]
    lo, hi = -14.0, 14.0
    x0, x1 = PAD + 8, W - PAD - 240
    y0, y1 = PAD, H - PAD

    def sy(v):
        return y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial">',
         f'<rect width="{W}" height="{H}" fill="#0e1216"/>',
         f'<text x="{PAD}" y="30" fill="#e7edf3" font-size="16" font-weight="600">'
         f'Balance margin — centre of mass vs the foot support polygon</text>',
         f'<text x="{PAD}" y="50" fill="#9aa7b4" font-size="12">'
         f'above the line the pose can be held still; below it needs momentum the clip does not have</text>']

    for v in range(-12, 13, 4):
        y = sy(v)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                 f'stroke="#1b2430" stroke-width="1"/>')
        p.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" fill="#5d6b7a" font-size="11" '
                 f'text-anchor="end">{v:+d} cm</text>')
    yz = sy(0)
    p.append(f'<line x1="{x0}" y1="{yz:.1f}" x2="{x1}" y2="{yz:.1f}" '
             f'stroke="#e0644a" stroke-width="1.5" stroke-dasharray="5 4"/>')

    for i, (lab, r, col) in enumerate(rows):
        m = r["margin"] * 100
        m = np.where(np.isfinite(m), m, lo)
        n = len(m)
        pts = " ".join(f"{x0 + k / (n - 1) * (x1 - x0):.1f},{sy(m[k]):.1f}"
                       for k in range(n))
        wide = 2.4 if lab.startswith(("A9", "BONES")) else 1.8
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                 f'stroke-width="{wide}" opacity="0.92"/>')
        ly = PAD + 6 + i * 46
        p.append(f'<rect x="{x1 + 24}" y="{ly - 10}" width="14" height="3" fill="{col}"/>')
        p.append(f'<text x="{x1 + 44}" y="{ly - 4}" fill="#e7edf3" font-size="12.5">{lab}</text>')
        p.append(f'<text x="{x1 + 44}" y="{ly + 13}" fill="#9aa7b4" font-size="11.5">'
                 f'{r["balanced"]:.0f}% balanced · knee/hip {r["knee_hip"]:.2f} · '
                 f'{r["both_down"]:.0f}% both feet down</text>')

    p.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 22}" fill="#5d6b7a" font-size="11.5" '
             f'text-anchor="middle">clip progress →</text>')
    p.append("</svg>")
    out = os.path.join(REPO := os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "fig_balance_profile.svg")
    open(out, "w").write("\n".join(p))
    print(f"wrote {out}")
    for lab, r, _ in rows:
        print(f"  {lab:32s} balanced {r['balanced']:3.0f}%  knee/hip {r['knee_hip']:.2f}  "
              f"median margin {r['margin_med']:+.1f} cm")


if __name__ == "__main__":
    main()
