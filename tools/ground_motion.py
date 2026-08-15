#!/usr/bin/env python3
"""Drop a BONES-SEED-format G1 CSV onto the floor.

GMR's IK matches joint positions but carries no ground-contact constraint, so a
retargeted clip can float. Measured on the Mixamo magic attack: 35.7-88.0 mm of
float, mean 53.6 mm, drifting 52 mm over the clip. A real BONES-SEED CSV sits at
-2.4..+0.5 mm by comparison — those were produced by the SOMA retargeter, which
does ground its output.

Uses MuJoCo's own collision detection against the floor plane and binary-searches
the vertical offset that brings the lowest robot geom to z=0.

    MUJOCO_GL=egl .venv/bin/python tools/ground_motion.py in.csv out.csv --mode perframe

Modes:
  perframe  every frame's lowest point is placed on the floor. Correct for a
            standing move with no flight phase (our hadouken). Removes the float
            entirely — and with it the ~51 mm of apparent hip drop that was
            float artefact rather than real crouch.
  offset    subtract one constant (the clip's minimum float) so nothing ever
            penetrates but relative vertical motion is untouched. Use this when
            the clip genuinely leaves the ground (jumps, the uppercut's drive).
"""

import argparse
import os

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def floor_geoms(model):
    return [g for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]


def measure_float(model, data, floor, root, quat, dof):
    """Per-frame height of the lowest robot geom above the floor, in metres."""
    out = np.empty(len(root))
    for i in range(len(root)):
        lo, hi = -0.40, 0.40
        for _ in range(44):
            mid = 0.5 * (lo + hi)
            data.qpos[:3] = root[i] + [0, 0, mid]
            data.qpos[3:7] = quat[i]
            data.qpos[7:] = dof[i]
            mujoco.mj_forward(model, data)
            gaps = [data.contact.dist[k] for k in range(data.ncon)
                    if data.contact.geom1[k] in floor or data.contact.geom2[k] in floor]
            if (min(gaps) if gaps else 1.0) > 0.0:
                hi = mid
            else:
                lo = mid
        out[i] = -hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--mode", choices=["perframe", "offset", "mean"], default="perframe")
    args = ap.parse_args()

    from general_motion_retargeting import params

    model = mujoco.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"]))
    data = mujoco.MjData(model)
    floor = floor_geoms(model)

    df = pd.read_csv(args.in_csv)
    jc = [c for c in df.columns if c.endswith("_dof")]
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values, degrees=True
    ).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)

    before = measure_float(model, data, floor, root, quat, dof)
    if args.mode == "perframe":
        # forces the lowest point onto the floor EVERY frame, which also drags the
        # body down whenever a foot legitimately lifts — it removed 18 cm of real
        # vertical range from A5
        shift = before
    elif args.mode == "offset":
        # never penetrates, but floats the whole clip by the worst frame (23 cm on A5)
        shift = np.full(len(before), before.min())
    else:
        # "mean": one constant shift, centred so float and penetration share the
        # error. Keeps the vertical dynamics intact, which is what the tracking
        # policy is actually asked to reproduce.
        shift = np.full(len(before), before.mean())

    df = df.copy()
    df["root_translateZ"] = df["root_translateZ"] - shift * 100.0  # metres -> cm

    root2 = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    after = measure_float(model, data, floor, root2, quat, dof)

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format="%.8f")
    print(f"mode={args.mode}")
    print(f"  float before (mm): min {before.min()*1000:7.1f}  max {before.max()*1000:7.1f}  "
          f"mean {before.mean()*1000:7.1f}  swing {(before.max()-before.min())*1000:6.1f}")
    print(f"  float after  (mm): min {after.min()*1000:7.1f}  max {after.max()*1000:7.1f}  "
          f"mean {after.mean()*1000:7.1f}  swing {(after.max()-after.min())*1000:6.1f}")
    print(f"  pelvis height (cm): {df.root_translateZ.min():.1f}..{df.root_translateZ.max():.1f} "
          f"(drop {df.root_translateZ.max()-df.root_translateZ.min():.1f})")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()


