#!/usr/bin/env python3
"""Export a G1 motion as per-body world transforms, plus its balance margin.

    MUJOCO_GL=egl .venv/bin/python tools/export_g1_bodies.py in.csv out.npz

Blender has no G1 rig, and the physics-track results live in G1 joint space rather
than as a human BVH. Rather than build a second rig, this runs the motion through
MuJoCo — the same model everything else is checked against — and writes out where
every body actually ends up each frame, along with the static balance margin.

Blender then only has to place boxes and keyframe them, so what is drawn is exactly
what was measured, with no second kinematics implementation to disagree.
"""

import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from g1_columns import joint_cols, was_reordered

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
CONTACT_H = 0.02
IN, OUT = sys.argv[1], sys.argv[2]

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
foot_g = [g for g in range(model.ngeom)
          if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                model.geom_bodyid[g]) or "")]

df = pd.read_csv(IN)
jc = joint_cols(df)
root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
quat = Rotation.from_euler("xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
                           degrees=True).as_quat()[:, [3, 0, 1, 2]]
dof = np.deg2rad(df[jc].values)
T = len(df)

# every body except the world
bodies = [b for b in range(1, model.nbody)]
bnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]

pos = np.zeros((T, len(bodies), 3))
mat = np.zeros((T, len(bodies), 3, 3))
margin = np.zeros(T)
com = np.zeros((T, 3))

for i in range(T):
    data.qpos[:3] = root[i]
    data.qpos[3:7] = quat[i]
    data.qpos[7:] = dof[i]
    mujoco.mj_forward(model, data)
    for k, b in enumerate(bodies):
        pos[i, k] = data.xpos[b]
        mat[i, k] = data.xmat[b].reshape(3, 3)
    com[i] = data.subtree_com[0]

    pts = []
    for g in foot_g:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz_ in (-1, 1):
                    pts.append(p + R @ (np.array([sx, sy, sz_]) * sz))
    pts = np.array(pts)
    on = pts[pts[:, 2] < pts[:, 2].min() + CONTACT_H]
    m = -0.12
    if len(on) >= 3:
        try:
            hull = ConvexHull(on[:, :2])
            inside, best = True, np.inf
            for eq in hull.equations:
                d = float(eq[:2] @ com[i][:2] + eq[2])
                if d > 0:
                    inside = False
                best = min(best, abs(d))
            m = best if inside else -best
        except Exception:
            pass
    margin[i] = m

# a box per body, sized from its geoms, so Blender can draw it without the meshes
size = np.zeros((len(bodies), 3))
for k, b in enumerate(bodies):
    gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
    size[k] = np.max([model.geom_size[g] for g in gs], axis=0) if gs else [0.02, 0.02, 0.02]
    size[k] = np.clip(size[k], 0.018, 0.16)

np.savez_compressed(OUT, pos=pos, mat=mat, size=size, margin=margin, com=com,
                    names=np.array(bnames), fps=60.0)
print(f"{os.path.basename(IN)}: {T} frames, {len(bodies)} bodies, "
      f"margin {margin.min() * 100:+.1f}..{margin.max() * 100:+.1f} cm -> {OUT}")
