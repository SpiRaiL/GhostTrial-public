#!/usr/bin/env python3
"""Figure: which parts of the move the robot could stand still in.

    MUJOCO_GL=egl .venv/bin/python tools/fig_static.py "<report dir>/fig_static.svg"

Balance margin over the clip — signed distance from the centre of mass to the edge
of the foot support polygon. Above zero the robot can stop there and stand; below
zero it is falling, and the only way it stays up is to take a step.
"""
import contextlib, io, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import static_feasibility as SF

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_static.svg"
BG, INK, DIM, GRID = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
GOOD, BAD, ACC = "#4ec9a3", "#e0736b", "#59b0ff"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

m = mujoco.MjModel.from_xml_path(SF.XML); d = mujoco.MjData(m)
floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
feet = [g for g in range(m.ngeom) if "ankle_roll" in
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = SF.analyse("data/gemx_g1_retimed/A4_natural_speed.csv", m, d, floor, feet)
mg = r["margin"] * 100
mg = np.where(np.isfinite(mg), mg, -12.0)      # no contact patch at all -> off the bottom
T = len(mg)
BEATS = [(30,"idle"),(120,"wind-up"),(205,"SPEAR THROW"),(300,"pull back"),
         (370,"guard"),(470,"crouch"),(520,"drive"),(600,"UPPERCUT"),(700,"reset")]

W,H = 1060,430; L,R,TP,B = 62,26,104,74
YLO,YHI = -12.0,10.0
X = lambda i: L + i/T*(W-L-R)
Y = lambda v: H-B - (v-YLO)/(YHI-YLO)*(H-B-TP)
p=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
   f'font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif">'
   f'<rect width="{W}" height="{H}" fill="{BG}"/>',
   f'<text x="24" y="30" fill="{INK}" font-size="15" font-weight="600">The robot cannot stand still in most of this move</text>',
   f'<text x="24" y="50" fill="{DIM}" font-size="12">balance margin — how far the centre of mass sits inside the foot support polygon · take A4, every frame</text>',
   f'<text x="24" y="70" fill="{DIM}" font-size="11.5">above the line the pose can be held; below it the robot is falling and must step to stay up. The move is slow enough that every frame should be holdable.</text>']
p.append(f'<rect x="{L}" y="{Y(YHI):.0f}" width="{W-L-R}" height="{Y(0)-Y(YHI):.0f}" fill="#12211c"/>')
p.append(f'<rect x="{L}" y="{Y(0):.0f}" width="{W-L-R}" height="{Y(YLO)-Y(0):.0f}" fill="#241618"/>')
for v in range(-12,11,4):
    p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
    p.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" fill="{DIM}" font-size="10.5" text-anchor="end" font-family="{MONO}">{v}</text>')
p.append(f'<text x="{L-8}" y="{TP-12}" fill="{DIM}" font-size="11" text-anchor="end">cm</text>')
p.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{W-R}" y2="{Y(0):.1f}" stroke="{INK}" stroke-width="1.5"/>')
pts=" ".join(f"{X(i):.1f},{Y(max(min(v,YHI),YLO)):.1f}" for i,v in enumerate(mg))
p.append(f'<polyline points="{pts}" fill="none" stroke="{ACC}" stroke-width="1.6"/>')
for f,lab in BEATS:
    if f>=T: continue
    v=mg[f]; col=GOOD if v>0 else BAD
    p.append(f'<line x1="{X(f):.1f}" y1="{TP}" x2="{X(f):.1f}" y2="{H-B}" stroke="{GRID}" stroke-width="1" stroke-dasharray="3 3"/>')
    p.append(f'<circle cx="{X(f):.1f}" cy="{Y(max(min(v,YHI),YLO)):.1f}" r="4.5" fill="{col}" stroke="{BG}" stroke-width="1.5"/>')
    yy = H-B+16 + (14 if BEATS.index((f,lab))%2 else 0)
    p.append(f'<text x="{X(f):.1f}" y="{yy}" fill="{col}" font-size="10" text-anchor="middle">{lab}</text>')
hold=int((r["margin"]>0).sum())
p.append(f'<text x="24" y="{H-8}" fill="{DIM}" font-size="11">only {hold} of {T} frames ({100*hold/T:.0f}%) are statically holdable · for comparison, BONES-SEED motions that SONIC tracks well sit at +2 to +7 cm median</text>')
p.append("</svg>")
open(OUT,"w").write("\n".join(p))
print("wrote",OUT)
