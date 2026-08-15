#!/usr/bin/env python3
"""Generate the report's SVG figures. Dark surface, house palette.

    MUJOCO_GL=egl .venv/bin/python tools/make_report_figs.py "<report dir>"

Palette is the report library's: accent #59b0ff, good #4ec9a3, bad #e0736b,
warn #e6b34d on #0f1216. Validated for CVD separation (worst adjacent pair
deltaE 8.6 deutan), normal-vision separation (17.1) and contrast (all >=3:1).
Every series is also direct-labelled, so identity is never colour-alone.
"""

import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

OUT = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

BG, INK, DIM, LINE = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
ACCENT, GOOD, BAD, WARN = "#59b0ff", "#4ec9a3", "#e0736b", "#e6b34d"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

from general_motion_retargeting import params  # noqa: E402

MODEL = mujoco.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"]))
DATA = mujoco.MjData(MODEL)
FLOOR = [g for g in range(MODEL.ngeom)
         if MODEL.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]


def load(csv, n=None):
    df = pd.read_csv(csv)
    df = df if n is None else df.iloc[:n]
    jc = [c for c in df.columns if c.endswith("_dof")]
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values, degrees=True
    ).as_quat()[:, [3, 0, 1, 2]]
    return df, root, quat, np.deg2rad(df[jc].values)


def floats(root, quat, dof):
    out = np.empty(len(root))
    for i in range(len(root)):
        lo, hi = -0.4, 0.4
        for _ in range(44):
            mid = 0.5 * (lo + hi)
            DATA.qpos[:3] = root[i] + [0, 0, mid]
            DATA.qpos[3:7] = quat[i]
            DATA.qpos[7:] = dof[i]
            mujoco.mj_forward(MODEL, DATA)
            g = [DATA.contact.dist[k] for k in range(DATA.ncon)
                 if DATA.contact.geom1[k] in FLOOR or DATA.contact.geom2[k] in FLOOR]
            if (min(g) if g else 1.0) > 0.0:
                hi = mid
            else:
                lo = mid
        out[i] = -hi
    return out * 1000.0  # mm


def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>']


def txt(x, y, s, fill=INK, size=13, anchor="start", weight="400", mono=False):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam}>{s}</text>')


# ───────────────────────────── figure 1 · grounding ─────────────────────────────
def fig_grounding(path):
    _, r_raw, q_raw, d_raw = load(f"{REPO}/data/mixamo_csv/magic_attack_raw/magic_attack.csv")
    _, r_g, q_g, d_g = load(f"{REPO}/data/mixamo_csv/magic_attack/magic_attack.csv")
    _, r_b, q_b, d_b = load(
        f"{REPO}/retargeted/bones-seed/g1/csv/230509/shadow_boxing_R_001__A359.csv", 35)
    before, after, ref = floats(r_raw, q_raw, d_raw), floats(r_g, q_g, d_g), floats(r_b, q_b, d_b)

    W, H = 1000, 420
    L, R, T, B = 78, 210, 46, 52
    pw, ph = W - L - R, H - T - B
    ymin, ymax = -12.0, 62.0
    n = len(before)

    def X(i):
        return L + pw * i / (n - 1)

    def Y(v):
        return T + ph * (ymax - v) / (ymax - ymin)

    s = svg_open(W, H)
    s.append(txt(L, 24, "Retargeted clip floats above the floor — and the float drifts",
                 INK, 15, weight="600"))
    for gv in range(0, 61, 20):
        s.append(f'<line x1="{L}" y1="{Y(gv):.1f}" x2="{L+pw}" y2="{Y(gv):.1f}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
        s.append(txt(L - 10, Y(gv) + 4, f"{gv}", DIM, 11.5, "end", mono=True))
    s.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{L+pw}" y2="{Y(0):.1f}" '
             f'stroke="{DIM}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    s.append(txt(L - 10, Y(0) - 8, "floor", DIM, 11, "end"))
    s.append(txt(20, T + ph / 2, "float (mm)", DIM, 12, "middle",
                 ) .replace("<text", f'<text transform="rotate(-90 20 {T+ph/2})"'))

    for i in range(0, n, 6):
        s.append(txt(X(i), H - 26, f"{i}", DIM, 11.5, "middle", mono=True))
    s.append(txt(L + pw / 2, H - 8, "frame", DIM, 12, "middle"))

    # label y is nudged per series: `after` and `ref` both sit on zero and would collide
    spec = ((before, BAD, "before grounding", f"{before.min():.0f}–{before.max():.0f} mm", -14),
            (after, GOOD, "after grounding", "0.0 mm, every frame", -6),
            (ref, ACCENT, "BONES-SEED reference", f"{ref.min():.1f}–{ref.max():.1f} mm", 34))
    for series, colour, label, note, dy in spec:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.5" '
                 f'stroke-linejoin="round"/>')
        ye = Y(series[-1])
        s.append(f'<circle cx="{X(n-1):.1f}" cy="{ye:.1f}" r="4" fill="{colour}" '
                 f'stroke="{BG}" stroke-width="2"/>')
        ylab = ye + dy
        s.append(f'<line x1="{X(n-1)+5:.1f}" y1="{ye:.1f}" x2="{L+pw+9:.1f}" y2="{ylab-4:.1f}" '
                 f'stroke="{colour}" stroke-width="1" opacity="0.5"/>')
        s.append(txt(L + pw + 12, ylab, label, colour, 12.5, weight="600"))
        s.append(txt(L + pw + 12, ylab + 15, note, DIM, 11.5, mono=True))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))
    print("wrote", path)
    return before, after, ref


# ──────────────────────────── figure 2 · joint saturation ───────────────────────
def fig_saturation(path):
    df = pd.read_csv(f"{REPO}/data/mixamo_csv/magic_attack/magic_attack.csv")
    rows = []
    for i in range(1, MODEL.njnt):
        nm = mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.rad2deg(MODEL.jnt_range[i])
        v = df[nm + "_dof"]
        pinned = v.min() <= lo + 0.5 or v.max() >= hi - 0.5
        rows.append((nm, float(lo), float(hi), float(v.min()), float(v.max()), pinned))

    W = 1000
    rowh = 21
    T, L, R, B = 80, 250, 120, 40
    H = T + rowh * len(rows) + B
    pw = W - L - R
    amin, amax = -180.0, 170.0

    def X(v):
        return L + pw * (v - amin) / (amax - amin)

    s = svg_open(W, H)
    s.append(txt(24, 26, "All 29 DOFs are inside the G1's limits — but six are pinned against them",
                 INK, 15, weight="600"))
    s.append(txt(24, 46, "grey = joint's mechanical range · blue = range the motion actually uses · "
                         "amber = hard against a limit", DIM, 12))
    for gv in (-180, -90, 0, 90, 170):
        s.append(f'<line x1="{X(gv):.1f}" y1="{T-8}" x2="{X(gv):.1f}" y2="{T+rowh*len(rows)}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
        s.append(txt(X(gv), T - 14, f"{gv}°", DIM, 11, "middle", mono=True))

    for k, (nm, lo, hi, vmin, vmax, pinned) in enumerate(rows):
        y = T + k * rowh + rowh / 2
        col = WARN if pinned else ACCENT
        s.append(txt(L - 12, y + 4, nm.replace("_joint", ""), DIM if not pinned else INK,
                     11.5, "end", weight="600" if pinned else "400", mono=True))
        s.append(f'<rect x="{X(lo):.1f}" y="{y-6:.1f}" width="{max(X(hi)-X(lo),1):.1f}" '
                 f'height="12" rx="4" fill="#1b222b" stroke="{LINE}" stroke-width="1"/>')
        s.append(f'<rect x="{X(vmin):.1f}" y="{y-5:.1f}" width="{max(X(vmax)-X(vmin),2):.1f}" '
                 f'height="10" rx="4" fill="{col}"/>')
        if pinned:
            s.append(txt(L + pw + 12, y + 4, "pinned", WARN, 11, weight="600"))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))
    print("wrote", path)
    return sum(1 for r in rows if r[5])


# ───────────────────────────── figure 3 · phase timeline ────────────────────────
def fig_phases(path):
    _, root, quat, dof = load(f"{REPO}/data/mixamo_csv/magic_attack/magic_attack.csv")
    lh = MODEL.body("left_wrist_yaw_link").id
    rh = MODEL.body("right_wrist_yaw_link").id
    mids, pelvis = [], []
    for i in range(len(root)):
        DATA.qpos[:3] = root[i]
        DATA.qpos[3:7] = quat[i]
        DATA.qpos[7:] = dof[i]
        mujoco.mj_forward(MODEL, DATA)
        mids.append((DATA.xpos[lh] + DATA.xpos[rh]) / 2)
        pelvis.append(root[i][2])
    mids, pelvis = np.array(mids), np.array(pelvis)
    speed = np.r_[0.0, np.linalg.norm(np.diff(mids, axis=0), axis=1) * 30.0]
    reach = mids[:, 0] - root[:, 0]

    W, H = 1000, 400
    L, R, T, B = 70, 190, 56, 52
    pw, ph = W - L - R, H - T - B
    n = len(speed)
    smax = max(6.0, speed.max() * 1.15)

    def X(i):
        return L + pw * i / (n - 1)

    def Y(v, lo, hi):
        return T + ph * (hi - v) / (hi - lo)

    s = svg_open(W, H)
    s.append(txt(24, 26, "The G1's version of the strike — 35 frames, 1.17 s", INK, 15, weight="600"))
    s.append(txt(365, 26, "hand speed is the two-wrist midpoint in world space", DIM, 12))

    # boundaries taken from the measured curves, not guessed: pelvis falls to
    # frame ~11, hand speed spikes 11-15, everything flat after
    bands = [(0, 5, "wind-up", "#171c24"), (5, 11, "charge", "#1d2530"),
             (11, 15, "thrust", "#24303d"), (15, n - 1, "hold", "#171c24")]
    for a, b, lab, fill in bands:
        s.append(f'<rect x="{X(a):.1f}" y="{T}" width="{X(b)-X(a):.1f}" height="{ph}" fill="{fill}"/>')
        s.append(txt((X(a) + X(b)) / 2, T + 16, lab, DIM, 11.5, "middle", weight="600"))

    for gv in (0, 2, 4, 6):
        yy = Y(gv, 0, smax)
        s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{LINE}" stroke-width="1"/>')
        s.append(txt(L - 10, yy + 4, f"{gv}", DIM, 11.5, "end", mono=True))
    s.append(txt(L - 10, T - 8, "m/s", DIM, 11, "end"))

    pts = " ".join(f"{X(i):.1f},{Y(v,0,smax):.1f}" for i, v in enumerate(speed))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{ACCENT}" stroke-width="2.5"/>')
    s.append(txt(X(n - 1) + 12, Y(speed[-1], 0, smax) + 4, "hand speed", ACCENT, 12.5, weight="600"))

    plo, phi = pelvis.min() - 0.02, pelvis.max() + 0.02
    pp = " ".join(f"{X(i):.1f},{Y(v,plo,phi):.1f}" for i, v in enumerate(pelvis))
    s.append(f'<polyline points="{pp}" fill="none" stroke="{GOOD}" stroke-width="2.5" stroke-dasharray="6 4"/>')
    s.append(txt(X(n - 1) + 12, Y(pelvis[-1], plo, phi) + 4, "pelvis height", GOOD, 12.5, weight="600"))
    s.append(txt(X(n - 1) + 12, Y(pelvis[-1], plo, phi) + 21,
                 f"{pelvis.max()*100:.0f}→{pelvis.min()*100:.0f} cm", DIM, 11.5, mono=True))

    pk = int(speed.argmax())
    s.append(f'<line x1="{X(pk):.1f}" y1="{T}" x2="{X(pk):.1f}" y2="{T+ph}" stroke="{WARN}" '
             f'stroke-width="1.5" stroke-dasharray="3 3"/>')
    s.append(txt(X(pk) + 8, T + 40, f"peak {speed.max():.1f} m/s", WARN, 11.5, weight="600"))

    for i in range(0, n, 5):
        s.append(txt(X(i), H - 26, f"{i}", DIM, 11.5, "middle", mono=True))
    s.append(txt(L + pw / 2, H - 8, "frame  (30 fps)", DIM, 12, "middle"))
    s.append("</svg>")
    open(path, "w").write("\n".join(s))
    print("wrote", path)
    return speed, reach


fig_grounding(f"{OUT}/fig_grounding.svg")
fig_saturation(f"{OUT}/fig_saturation.svg")
fig_phases(f"{OUT}/fig_phases.svg")
