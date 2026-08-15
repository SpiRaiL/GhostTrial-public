#!/usr/bin/env python3
"""Feasibility judged against the contact schedule the motion actually has.

    MUJOCO_GL=egl .venv/bin/python tools/contact_feasibility.py <csv> [...]

The earlier check (tools/static_feasibility.py) asked whether the centre of mass
sat inside the points that happened to be within 2 cm of the ground. That is two
assumptions wrong at once:

  * it treated single support as failure. If the performer is genuinely standing on
    one foot, the pose is fine — it just has a smaller polygon. Stepping is a
    legitimate way to balance and is not a defect.
  * it treated a TILTED foot as having almost no support, because only the leading
    corner was near the ground. In reality a foot bearing load settles flat, so the
    support available is the whole sole, not the corner that happens to touch in a
    kinematic pose.

So contact is detected the way the imitation literature does it — a foot is in
contact when it is both LOW and SLOW — and the support polygon is the union of the
full soles of whichever feet are in contact. Frames with no foot in contact are
flight, and are reported separately rather than counted as balance failures.
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
H_THRESH = 0.045          # metres: "low"
V_THRESH = 0.35           # m/s: "slow"


def foot_corners(model, data, geoms):
    pts = []
    for g in geoms:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz_ in (-1, 1):
                    pts.append(p + R @ (np.array([sx, sy, sz_]) * sz))
    return np.array(pts)


def margin_in(poly, com_xy):
    if len(poly) < 3:
        return -np.inf
    try:
        hull = ConvexHull(poly)
    except Exception:
        return -np.inf
    inside, best = True, np.inf
    for eq in hull.equations:
        d = float(eq[:2] @ com_xy + eq[2])
        if d > 0:
            inside = False
        best = min(best, abs(d))
    return best if inside else -best


def analyse(path, model, data, side_geoms, fps=60.0):
    df = pd.read_csv(path)
    jc = joint_cols(df)
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    T = len(df)

    low = np.zeros((T, 2))
    cen = np.zeros((T, 2, 3))
    corners = [None] * T
    com = np.zeros((T, 3))
    for i in range(T):
        data.qpos[:3] = root[i]
        data.qpos[3:7] = quat[i]
        data.qpos[7:] = dof[i]
        mujoco.mj_forward(model, data)
        com[i] = data.subtree_com[0]
        cs = []
        for s, gs in enumerate(side_geoms):
            c = foot_corners(model, data, gs)
            cs.append(c)
            low[i, s] = c[:, 2].min()
            cen[i, s] = c.mean(axis=0)
        corners[i] = cs

    floor = np.percentile(low.min(axis=1), 2)
    speed = np.zeros((T, 2))
    speed[1:] = np.linalg.norm(np.diff(cen, axis=0), axis=2) * fps
    speed[0] = speed[1]
    contact = ((low - floor) < H_THRESH) & (speed < V_THRESH)

    margin = np.full(T, -np.inf)
    for i in range(T):
        poly = []
        for s in range(2):
            if contact[i, s]:
                # the WHOLE sole, flattened onto the floor — a loaded foot settles
                poly.append(corners[i][s][:, :2])
        if poly:
            margin[i] = margin_in(np.vstack(poly), com[i][:2])

    both = contact.all(axis=1)
    one = contact.sum(axis=1) == 1
    none = contact.sum(axis=1) == 0
    held = margin > 0

    print(f"\n\033[1m{os.path.basename(path)}\033[0m  {T} frames")
    print(f"  contact schedule   double {100 * both.mean():3.0f}%   "
          f"single {100 * one.mean():3.0f}%   flight {100 * none.mean():3.0f}%")
    fin = margin[np.isfinite(margin)]
    if len(fin):
        print(f"  balance margin     median {np.median(fin) * 100:+6.1f} cm   "
              f"worst {fin.min() * 100:+6.1f} cm")
    print(f"  \033[1mbalanced against its own support   {100 * held.mean():3.0f}%"
          f"  ({int(held.sum())}/{T})\033[0m")
    if none.any():
        print(f"  ...of the rest, {int((none).sum())} are flight frames "
              f"(no foot down) and {int((~held & ~none).sum())} are genuinely off balance")
    return dict(margin=margin, contact=contact, held=held)


def main(paths):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    side_geoms = []
    for side in ("left", "right"):
        side_geoms.append([g for g in range(model.ngeom)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                 model.geom_bodyid[g]) or "")
                           == f"{side}_ankle_roll_link"])
    print(f"contact: foot low (<{H_THRESH * 100:.0f} cm) AND slow (<{V_THRESH} m/s)")
    return {p: analyse(p, model, data, side_geoms) for p in paths}


if __name__ == "__main__":
    main(sys.argv[1:])
