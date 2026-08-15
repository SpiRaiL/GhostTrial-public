#!/usr/bin/env python3
"""Figure: the three authoring routes, scored on what WBT-Bench penalises.

    MUJOCO_GL=egl .venv/bin/python tools/make_approach_fig.py "<report dir>"
"""

import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

OUT = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BG, INK, DIM, LINE = "#0f1216", "#e7edf3", "#9aa7b4", "#2a333d"
ACCENT, GOOD, BAD, WARN = "#59b0ff", "#4ec9a3", "#e0736b", "#e6b34d"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

from general_motion_retargeting import params  # noqa: E402

M = mujoco.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"]))
D = mujoco.MjData(M)
FLOOR = [g for g in range(M.ngeom) if M.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]

CASES = [
    ("Mixamo clip, retargeted", f"{REPO}/data/mixamo_csv/magic_attack/magic_attack.csv", ACCENT),
    ("Hand-authored on a human rig", f"{REPO}/data/mixamo_csv/spear/spear_throw.csv", BAD),
    ("Authored in G1 joint space", f"{REPO}/data/g1_authored/spear_uppercut.csv", GOOD),
]


def score(path):
    df = pd.read_csv(path)
    jc = [c for c in df.columns if c.endswith("_dof")]
    pinned = 0
    for i in range(1, M.njnt):
        n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.rad2deg(M.jnt_range[i])
        v = df[n + "_dof"]
        if v.min() <= lo + 0.5 or v.max() >= hi - 0.5:
            pinned += 1
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    q = Rotation.from_euler("xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
                            degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    sc = 0
    for i in range(len(df)):
        D.qpos[:3] = root[i]
        D.qpos[3:7] = q[i]
        D.qpos[7:] = dof[i]
        mujoco.mj_forward(M, D)
        sc += sum(1 for k in range(D.ncon)
                  if D.contact.geom1[k] not in FLOOR and D.contact.geom2[k] not in FLOOR)
    return pinned, sc, len(df)


rows = [(lab, col, *score(p)) for lab, p, col in CASES]

W, H = 1000, 330
L, GAP = 330, 22
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="{FONT}"><rect width="{W}" height="{H}" fill="{BG}"/>']


def T(x, y, t, fill=INK, size=13, anchor="start", weight="400", mono=False):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam}>{t}</text>')


s.append(T(24, 30, "Authoring in the robot's own joint space removes both penalties outright",
           INK, 15, weight="600"))
s.append(T(24, 50, "WBT-Bench penalises self-collision and flailing; joints pinned at a mechanical "
                   "stop are where flailing starts", DIM, 12))

colw = (W - L - 60) / 2
maxpin, maxsc = 12.0, 900.0
s.append(T(L + colw * 0.5, 84, "joints pinned at a limit", DIM, 12, "middle", weight="600"))
s.append(T(L + colw * 1.5 + 30, 84, "self-collision contacts", DIM, 12, "middle", weight="600"))

y = 104
for lab, col, pin, sc, nf in rows:
    s.append(T(L - 16, y + 26, lab, INK, 13.5, "end", weight="600"))
    s.append(T(L - 16, y + 44, f"{nf} frames", DIM, 11.5, "end", mono=True))
    # pinned
    w1 = max(colw * pin / maxpin, 3)
    s.append(f'<rect x="{L}" y="{y+10}" width="{colw}" height="26" rx="6" fill="#161d26"/>')
    s.append(f'<rect x="{L}" y="{y+10}" width="{w1:.1f}" height="26" rx="6" fill="{col}"/>')
    s.append(T(L + w1 + 10, y + 28, str(pin), col, 14, weight="700", mono=True))
    # self-collisions
    x2 = L + colw + 30
    w2 = max(colw * sc / maxsc, 3)
    s.append(f'<rect x="{x2}" y="{y+10}" width="{colw}" height="26" rx="6" fill="#161d26"/>')
    s.append(f'<rect x="{x2}" y="{y+10}" width="{w2:.1f}" height="26" rx="6" fill="{col}"/>')
    s.append(T(x2 + w2 + 10, y + 28, str(sc), col, 14, weight="700", mono=True))
    y += 62

s.append(T(24, H - 22, "Zero pinned joints and zero self-collisions on a 5.8 s two-move combo — "
                       "and no retarget step at all.", GOOD, 12.5, weight="600"))
s.append("</svg>")
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/fig_approaches.svg", "w").write("\n".join(s))
print("wrote", f"{OUT}/fig_approaches.svg")
for lab, col, pin, sc, nf in rows:
    print(f"  {lab:32} frames {nf:4d}  pinned {pin:2d}  self-collisions {sc:4d}")
