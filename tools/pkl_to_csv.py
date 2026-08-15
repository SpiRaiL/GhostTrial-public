#!/usr/bin/env python3
"""motion_lib .pkl -> BONES-SEED CSV, so a target renders through the same path as a rollout.

    .venv/bin/python tools/pkl_to_csv.py data/motion_lib_capture/robot/b1/B1_idle.pkl out.csv

Target and policy have to be drawn by the same code or the comparison is not one.
Columns come out in MuJoCo joint order (see tools/g1_columns.py), degrees and
centimetres, matching what every other tool here reads.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_columns import mujoco_joint_names  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
d = joblib.load(src)
key = list(d)[0]
e = d[key]

dof = np.asarray(e["dof"])
root = np.asarray(e["root_trans_offset"])
rot = np.asarray(e["root_rot"])
eul = Rotation.from_quat(rot).as_euler("xyz", degrees=True)   # pkl stores xyzw

names = mujoco_joint_names()
if dof.shape[1] != len(names):
    raise SystemExit(f"{dof.shape[1]} dof in pkl, {len(names)} joints in the model")

out = pd.DataFrame({
    "root_translateX": root[:, 0] * 100.0,
    "root_translateY": root[:, 1] * 100.0,
    "root_translateZ": root[:, 2] * 100.0,
    "root_rotateX": eul[:, 0], "root_rotateY": eul[:, 1], "root_rotateZ": eul[:, 2],
})
for i, n in enumerate(names):
    out[f"{n}_dof"] = np.rad2deg(dof[:, i])

out.to_csv(dst, index=False, float_format="%.6f")
print(f"{key}: {len(out)} frames @ {e['fps']} fps -> {dst}")
