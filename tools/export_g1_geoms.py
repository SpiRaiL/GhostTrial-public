#!/usr/bin/env python3
"""Emit the G1's visual geoms — mesh, parent body, local transform, colour — as JSON.

    MUJOCO_GL=egl .venv/bin/python tools/export_g1_geoms.py out.json

tools/export_g1_bodies.py gives where each BODY goes. That is enough to place
boxes, and not enough to build the robot: the model has 59 mesh geoms across 30
bodies, so most bodies carry several meshes, each with its own offset inside the
body. torso_link alone has six — including the HEAD, which is not a body of its own
and disappears entirely if meshes are matched to bodies one for one.

Colour comes from the geom too. The G1 is two-tone, 0.2 grey shells over 0.7 grey
structure, and collapsing that into a single material is what made the first render
look like a blank mannequin.
"""

import json
import os
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
MESHDIR = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                       "assets", "robot_description", "meshes", "g1")

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data", "g1_geoms.json")
model = mujoco.MjModel.from_xml_path(XML)

geoms = []
for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
    stl = os.path.join(MESHDIR, f"{mesh}.STL")
    geoms.append(dict(
        mesh=mesh, body=body,
        stl=stl if os.path.exists(stl) else None,
        pos=[float(x) for x in model.geom_pos[g]],
        quat=[float(x) for x in model.geom_quat[g]],      # w, x, y, z
        rgba=[float(x) for x in model.geom_rgba[g]],
    ))

missing = [g["mesh"] for g in geoms if not g["stl"]]
json.dump(geoms, open(out_path, "w"), indent=1)
bodies = {g["body"] for g in geoms}
print(f"{len(geoms)} mesh geoms across {len(bodies)} bodies -> {out_path}")
print(f"  bodies carrying more than one mesh: "
      f"{sum(1 for b in bodies if sum(g['body'] == b for g in geoms) > 1)}")
print(f"  distinct colours: {sorted({tuple(g['rgba']) for g in geoms})}")
if missing:
    print(f"  WARNING no STL for: {sorted(set(missing))}")
