#!/usr/bin/env python3
"""Render the 29-DoF body track together with its 14-DoF Dex3 hand track.

    MUJOCO_GL=egl .venv/bin/python tools/render_with_hands.py body.csv hands.csv out.mp4

The body comes from the trained policy (or its target) and the hands from
tools/author_hands.py. They are separate tracks because that is how the robot runs
them — SONIC drives 29 joints and the Dex3 hands take their own position commands —
so this is the first render that shows what the whole machine actually does.

Joint columns are matched BY NAME into the 43-joint model's own order. The body and
hand models number their joints differently, and indexing by position across two
models is the same mistake that made every policy video wrong for days.
"""

import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the scene_ variant adds the floor and lights; the bare model renders on black
HAND_XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic_deploy",
                        "g1", "scene_29dof_with_hand.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body_csv")
    ap.add_argument("hands_csv")
    ap.add_argument("out_mp4")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=315.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=3.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(HAND_XML)
    # This XML declares no visual/global offwidth, so the offscreen buffer defaults
    # smaller than the frame we ask for and MuJoCo refuses to render. Set it here
    # rather than editing a vendored asset.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    order = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(1, model.njnt)]

    body = pd.read_csv(args.body_csv)
    hands = pd.read_csv(args.hands_csv)
    have = {}
    for src in (body, hands):
        for c in src.columns:
            if c.endswith("_dof"):
                have[c[:-4]] = src[c].values

    missing = [n for n in order if n not in have]
    if missing:
        raise SystemExit(f"no column for {len(missing)} joints, e.g. {missing[:4]}")

    n = min(len(body), len(hands))
    q = np.deg2rad(np.column_stack([have[nme][:n] for nme in order]))
    root = body[["root_translateX", "root_translateY", "root_translateZ"]].values[:n] / 100.0
    quat = Rotation.from_euler(
        "xyz", body[["root_rotateX", "root_rotateY", "root_rotateZ"]].values[:n],
        degrees=True).as_quat()[:, [3, 0, 1, 2]]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = [root[:, 0].mean(), root[:, 1].mean(), 0.8]

    with mujoco.Renderer(model, args.height, args.width) as r, \
            imageio.get_writer(args.out_mp4, fps=args.fps, macro_block_size=1) as w:
        for i in range(n):
            data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], q[i]
            mujoco.mj_forward(model, data)
            r.update_scene(data, cam)
            w.append_data(r.render())

    print(f"{n} frames, {len(order)} joints ({len(order) - 29} of them hand) -> {args.out_mp4}")


if __name__ == "__main__":
    main()
