#!/usr/bin/env python3
"""Can the G1 hold this pose standing still? Checked frame by frame, no physics run.

    MUJOCO_GL=egl .venv/bin/python tools/static_feasibility.py data/gemx_g1_retimed/*.csv

The move is slow enough to be quasi-static: the robot should be able to stop on
any frame and stand there. That makes every frame a static-balance problem, and a
static problem can be answered exactly, in milliseconds, with no policy and no GPU.

Four questions per frame:

  limits          is every joint inside its mechanical range?
  self-collision  is the robot intersecting itself?
  support         is there a foot contact patch at all?
  balance         does the centre of mass project inside that patch, and by how
                  much? This is the one that decides whether a pose can be held.
                  A negative margin means the robot is falling, whatever the
                  tracking reward says.

The margin is signed distance from the CoM's ground projection to the edge of the
support polygon — positive inside, negative outside — so it reads directly as
"centimetres from toppling".
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
PIN = 0.02
CONTACT_H = 0.02          # metres: a foot point this close to the floor is bearing


def poly_margin(pts, p):
    """Signed distance from p to the convex hull of pts. Positive = inside."""
    if len(pts) < 3:
        return -np.inf
    try:
        hull = ConvexHull(pts)
    except Exception:
        return -np.inf
    best = np.inf
    inside = True
    for eq in hull.equations:                 # a·x + b ≤ 0 inside
        d = float(eq[:2] @ p + eq[2])
        if d > 0:
            inside = False
        best = min(best, abs(d))
    return best if inside else -best


def analyse(path, model, data, floor_geoms, foot_geoms):
    df = pd.read_csv(path)
    jc = joint_cols(df)
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    T = len(df)

    margin = np.full(T, -np.inf)
    selfcol = np.zeros(T)
    viol = ((dof < lo - 1e-6) | (dof > hi + 1e-6)).any(axis=1)
    npts = np.zeros(T)

    for i in range(T):
        data.qpos[:3] = root[i]
        data.qpos[3:7] = quat[i]
        data.qpos[7:] = dof[i]
        mujoco.mj_forward(model, data)

        for c in data.contact[:data.ncon]:
            if c.geom1 not in floor_geoms and c.geom2 not in floor_geoms:
                selfcol[i] += 1

        # support patch: foot geom contact points near the ground
        pts = []
        for g in foot_geoms:
            # sample the geom's own frame corners, which is enough for a foot box
            pos = data.geom_xpos[g]
            R = data.geom_xmat[g].reshape(3, 3)
            sz = model.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz_ in (-1, 1):
                        p = pos + R @ (np.array([sx, sy, sz_]) * sz)
                        pts.append(p)
        pts = np.array(pts)
        if len(pts):
            ground = pts[:, 2].min()
            on = pts[pts[:, 2] < ground + CONTACT_H]
            npts[i] = len(on)
            if len(on) >= 3:
                com = data.subtree_com[0]
                margin[i] = poly_margin(on[:, :2], com[:2])

    ok = (margin > 0) & (selfcol == 0) & (~viol)
    print(f"\n\033[1m{os.path.basename(path)}\033[0m  {T} frames")
    print(f"  joint-limit violations   {int(viol.sum()):5d} frames")
    print(f"  self-collision           {int((selfcol > 0).sum()):5d} frames")
    print(f"  balance margin (cm)      median {np.median(margin[np.isfinite(margin)]) * 100:6.1f}"
          f"   worst {margin[np.isfinite(margin)].min() * 100:6.1f}")
    print(f"  frames CoM outside foot support   {int((margin <= 0).sum()):5d}"
          f"  ({100 * (margin <= 0).mean():.0f}%)")
    print(f"  \033[1mstatically holdable          {int(ok.sum()):5d} / {T}"
          f"  ({100 * ok.mean():.0f}%)\033[0m")
    return dict(margin=margin, ok=ok, T=T)


def main(paths):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    floor = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    feet = [g for g in range(model.ngeom)
            if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                  model.geom_bodyid[g]) or "")]
    print(f"model {os.path.basename(XML)}  ({len(feet)} foot geoms)")
    return {p: analyse(p, model, data, floor, feet) for p in paths}


if __name__ == "__main__":
    main(sys.argv[1:])
