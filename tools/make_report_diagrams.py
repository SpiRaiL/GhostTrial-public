#!/usr/bin/env python3
"""Hand-authored SVG flow diagrams for the progress report (pipeline + plan)."""

import os
import sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

BG, INK, DIM, LINE = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
ACCENT, GOOD, BAD, WARN = "#59b0ff", "#4ec9a3", "#e0736b", "#e6b34d"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def head(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>',
            '<defs>'
            f'<marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{DIM}"/></marker>'
            f'<marker id="ag" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{GOOD}"/></marker>'
            '</defs>']


def T(x, y, s, fill=INK, size=13, anchor="start", weight="400", mono=False, op=1.0):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" opacity="{op}"{fam}>{s}</text>')


def box(x, y, w, h, title, sub=None, colour=ACCENT, note=None, dashed=False):
    d = ' stroke-dasharray="6 4"' if dashed else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#151b23" '
           f'stroke="{colour}" stroke-width="1.8"{d}/>',
           f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{colour}"/>']
    out.append(T(x + 16, y + 24, title, INK, 13.5, weight="600"))
    if sub:
        out.append(T(x + 16, y + 43, sub, DIM, 11.5, mono=True))
    if note:
        out.append(T(x + 16, y + 61, note, colour, 11.5, weight="600"))
    return out


def arrow(x1, y1, x2, y2, colour=DIM, marker="a", label=None):
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
           f'stroke-width="1.6" marker-end="url(#{marker})"/>']
    if label:
        out.append(T((x1 + x2) / 2, (y1 + y2) / 2 - 7, label, DIM, 11, "middle"))
    return out


# ─────────────────────────────── pipeline diagram ───────────────────────────────
W, H = 1000, 610
s = head(W, H)
s.append(T(24, 30, "The magic-attack pipeline — everything left of the dashed line runs locally",
           INK, 15, weight="600"))
s.append(T(24, 50, "green = built and verified today · amber = next up · red = blocked on budget/scale",
           DIM, 12))

BX, BW, BH = 40, 250, 78
ys = [80, 178, 276, 374]
s += box(BX, ys[0], BW, BH, "Mixamo FBX", "standing_2h_magic_attack_04.fbx", GOOD,
         "65-joint mixamorig, no skin")
s += box(BX, ys[1], BW, BH, "BVH  (centimetres)", "tools/fbx_to_bvh.py", GOOD,
         "trimmed f18–52, 35 frames")
s += box(BX, ys[2], BW, BH, "G1 29-DoF CSV", "tools/bvh_to_bones_csv.py", GOOD,
         "GMR --format nokov")
s += box(BX, ys[3], BW, BH, "grounded CSV", "tools/ground_motion.py", GOOD,
         "float 34.5 mm → 0.0 mm")
for a, b in zip(ys, ys[1:]):
    s += arrow(BX + BW / 2, a + BH, BX + BW / 2, b - 6, GOOD, "ag")

MX = 360
s += box(MX, ys[2], BW, BH, "motion_lib PKL", "convert_soma_csv_to_motion_lib.py", GOOD,
         "NVIDIA's own converter")
s += arrow(BX + BW, ys[3] + BH / 2, MX + BW / 2 - 6, ys[3] + BH / 2, GOOD, "ag")
s.append(f'<line x1="{MX+BW/2}" y1="{ys[3]+BH/2}" x2="{MX+BW/2}" y2="{ys[2]+BH+6}" '
         f'stroke="{GOOD}" stroke-width="1.6" marker-end="url(#ag)"/>')

s.append(f'<line x1="{MX+BW+30}" y1="70" x2="{MX+BW+30}" y2="{H-70}" stroke="{DIM}" '
         f'stroke-width="1.5" stroke-dasharray="7 5"/>')
s.append(T(MX + BW + 22, 64, "local RTX 5080", DIM, 11.5, "end", weight="600"))
s.append(T(MX + BW + 38, 64, "Nebius", DIM, 11.5, "start", weight="600"))

NX = MX + BW + 60
s += box(NX, ys[1], 250, BH, "SONIC fine-tune", "train_agent_trl.py", WARN,
         "+checkpoint=sonic_release", dashed=True)
s += box(NX, ys[2], 250, BH, "ONNX export", "--export-onnx", WARN,
         "hackathon deliverable", dashed=True)
s += box(NX, ys[3], 250, BH, "before/after sim video", "local render, free", GOOD, dashed=True)
s += arrow(MX + BW, ys[2] + 20, NX - 6, ys[1] + BH / 2)
s += arrow(NX + 125, ys[1] + BH, NX + 125, ys[2] - 6)
s += arrow(NX + 125, ys[2] + BH, NX + 125, ys[3] - 6)

s += box(40, 470, 570, 66, "BONES-SEED  ·  142,220 motions, 49 GB extracted", None, BAD,
         "no uppercut, no hadouken — usable only as style/stance variation")
s += arrow(610, 495, MX + BW / 2 + 40, ys[2] + BH + 8, BAD, "a", "style variation only")
s.append("</svg>")
open(f"{OUT}/fig_pipeline.svg", "w").write("\n".join(s))
print("wrote", f"{OUT}/fig_pipeline.svg")

# ───────────────────────────────── plan diagram ─────────────────────────────────
W2, H2 = 1000, 520
s = head(W2, H2)
s.append(T(24, 30, "Plan to submission — 9 days, $25 of Nebius, one capture in flight", INK, 15, weight="600"))
s.append(T(24, 50, "same shape as the last challenge: prove everything on the local GPU, "
                   "then buy only the fine-tune. A is already done, so C is unblocked today.", DIM, 12))

steps = [
    ("A", "Authored combo — DONE", "spear throw + uppercut in G1 joint space\n"
     "0 pinned, 0 self-collisions, 0 violations", GOOD, "local · free · complete"),
    ("B", "Commissioned capture", "dancer films the phrase (Upwork brief)\n"
     "~$100 · needed in hand by Aug 11-12", ACCENT, "external · ~$100"),
    ("C", "Smoke-test training", "IsaacLab 2.3.2, num_envs=16, 1 GPU,\n"
     "on the AUTHORED clip - no need to wait for film", ACCENT, "local RTX 5080 · free"),
    ("D", "Fine-tune", "sonic_release checkpoint; mix in baseline\n"
     "locomotion or the fundamentals check suffers", WARN, "Nebius · ~$18"),
    ("E", "Export + record", "ONNX, before/after video, dataset docs", GOOD, "local · free"),
]
y = 80
for k, (letter, title, body, colour, cost) in enumerate(steps):
    h = 74
    s.append(f'<rect x="40" y="{y}" width="920" height="{h}" rx="9" fill="#151b23" '
             f'stroke="{LINE}" stroke-width="1.4"/>')
    s.append(f'<circle cx="72" cy="{y+h/2}" r="17" fill="{colour}" opacity="0.16"/>')
    s.append(f'<circle cx="72" cy="{y+h/2}" r="17" fill="none" stroke="{colour}" stroke-width="1.8"/>')
    s.append(T(72, y + h / 2 + 6, letter, colour, 15, "middle", weight="700"))
    s.append(T(104, y + 27, title, INK, 13.5, weight="600"))
    for j, ln in enumerate(body.split("\n")):
        s.append(T(104, y + 47 + j * 15, ln, DIM, 11.5, mono=True))
    s.append(T(944, y + 27, cost, colour, 12, "end", weight="600"))
    if k < len(steps) - 1:
        s.append(f'<line x1="72" y1="{y+h}" x2="72" y2="{y+h+14}" stroke="{LINE}" stroke-width="1.6"/>')
    y += h + 14
s.append("</svg>")
open(f"{OUT}/fig_plan.svg", "w").write("\n".join(s))
print("wrote", f"{OUT}/fig_plan.svg")
