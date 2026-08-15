#!/usr/bin/env python3
"""The gate before spending GPU money: is this motion trainable on the G1?

    MUJOCO_GL=egl .venv/bin/python tools/check_g1_motion.py data/gemx_g1_retimed/*.csv

Three things sink a SONIC run, and all three are visible kinematically, before any
policy exists:

  * joint-limit violations — the reference asks for a pose the robot cannot hold;
  * joints PINNED at a stop — legal, but zero control authority left, and when it
    is an ankle that is the balance margin gone;
  * self-collision — WBT-Bench penalises it directly, and BONES-SEED, the training
    distribution, sits at 0.00-0.01 contacts per frame.

Checked against SONIC's own g1_29dof_rev_1_0.xml, NOT GMR's g1_mocap_29dof.xml.
The two disagree on 15 of 29 joints and the GMR one is far tighter, so validating
against it reports violations that do not exist in the robot being trained.
"""

import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from g1_columns import joint_cols, was_reordered

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
PIN = 0.02          # rad of a limit; ~1.15 deg counts as pinned


def check(path, model, data, floor_geoms):
    df = pd.read_csv(path)
    jc = joint_cols(df)
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    T = len(df)

    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]   # skip the free joint
    viol = (dof < lo - 1e-6) | (dof > hi + 1e-6)
    pinned = (dof < lo + PIN) | (dof > hi - PIN)

    contacts, depth = np.zeros(T), np.zeros(T)
    for i in range(T):
        data.qpos[:3] = root[i]
        data.qpos[3:7] = quat[i]
        data.qpos[7:] = dof[i]
        mujoco.mj_forward(model, data)
        n, d = 0, 0.0
        for c in data.contact[:data.ncon]:
            if c.geom1 in floor_geoms or c.geom2 in floor_geoms:
                continue                                     # ground contact is wanted
            n += 1
            d = min(d, c.dist)
        contacts[i], depth[i] = n, -d

    print(f"\n\033[1m{os.path.basename(path)}\033[0m  {T} frames ({T / 60:.1f}s @60fps)")
    nv = int(viol.any(axis=1).sum())
    print(f"  joint-limit violations   {int(viol.sum()):5d} over {nv} frames"
          f"{'' if nv else '   (none)'}")
    if nv:
        for j in np.where(viol.any(axis=0))[0]:
            over = dof[viol[:, j], j]
            print(f"      {jc[j]:34s} {viol[:, j].sum():4d} frames, worst "
                  f"{np.rad2deg(max(over.max() - hi[j], lo[j] - over.min())):.1f} deg out")
    npj = np.where(pinned.any(axis=0))[0]
    print(f"  joints pinned at a stop  {len(npj):5d}"
          + (f"   [{', '.join(jc[j].replace('_joint_dof', '') for j in npj)}]" if len(npj) else ""))
    print(f"  self-collision           {contacts.mean():5.2f} contacts/frame "
          f"(max {int(contacts.max())}), deepest {depth.max() * 1000:.1f} mm")
    print(f"                           BONES-SEED training distribution: 0.00-0.01/frame")
    return dict(viol=int(viol.sum()), pinned=len(npj), contacts=contacts.mean(),
                depth=depth.max())


def main(paths):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    floor = {g for g in range(model.ngeom)
             if model.geom_bodyid[g] == 0}                   # anything on worldbody
    print(f"model: {os.path.basename(XML)}  ({model.nu} actuators, {model.ngeom} geoms)")
    out = {p: check(p, model, data, floor) for p in paths}
    worst = max(out.values(), key=lambda d: d["contacts"])
    print(f"\nworst self-collision across the set: {worst['contacts']:.2f} contacts/frame")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
