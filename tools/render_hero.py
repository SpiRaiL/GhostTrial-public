#!/usr/bin/env python3
"""Hero shot: the policy under physics, looped, with the camera flying around it.

    MUJOCO_GL=egl .venv/bin/python tools/render_hero.py \
        data/rollouts/t12_policy.csv reports/hero_t12.mp4 --loops 3

One robot, no side-by-side target — this is the submission clip, so it shows what
the machine does rather than how closely it matches a reference.

The camera orbits continuously through the whole render rather than once per loop,
so the angle keeps changing as the move repeats and no two passes look the same. It
also rises and falls slightly and breathes in and out on distance, because a
constant-radius turntable reads as a CAD viewer rather than a camera.

It tracks a SMOOTHED root position. Following the pelvis frame by frame transfers
every step and bob into the camera and makes the shot seasick; a long moving average
keeps the robot framed while the camera itself stays still.

Looping repeats the clip end to end. The policy's last pose is not its first, so
there is a visible cut at the seam — real, not a rendering artefact, and the reason
the loop count is small.
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
from g1_columns import joint_cols  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out_mp4")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--turns", type=float, default=1.25,
                    help="full camera revolutions across the whole render")
    ap.add_argument("--start-az", type=float, default=315.0,
                    help="opening angle; 315 is the front three-quarter used elsewhere")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)

    df = pd.read_csv(args.csv)
    jc = joint_cols(df)
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    T = len(df)
    N = T * args.loops

    # smooth the point the camera looks at, or every step shakes the shot
    w = max(3, int(1.5 * args.fps) | 1)
    pad = np.pad(root[:, :2], ((w // 2, w // 2), (0, 0)), mode="edge")
    look = np.stack([np.convolve(pad[:, k], np.ones(w) / w, mode="valid")
                     for k in range(2)], axis=1)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    with mujoco.Renderer(model, args.height, args.width) as r, \
            imageio.get_writer(args.out_mp4, fps=args.fps, macro_block_size=1,
                               quality=8) as wr:
        for k in range(N):
            i = k % T
            u = k / max(N - 1, 1)
            cam.azimuth = args.start_az + 360.0 * args.turns * u
            # ease down and back up rather than sitting at one height
            # Elevation stays in a narrow band: past about -20 the camera looks over
            # the top of the sky gradient and the upper frame goes black. Distance is
            # tighter than the side-by-side renders because there is only one robot to
            # fit, and it should fill the frame.
            cam.elevation = -9.0 - 6.0 * (0.5 - 0.5 * np.cos(2 * np.pi * u))
            cam.distance = 2.25 + 0.30 * np.sin(2 * np.pi * u)
            cam.lookat[:] = [look[i, 0], look[i, 1], 0.80]
            data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], dof[i]
            mujoco.mj_forward(model, data)
            r.update_scene(data, cam)
            wr.append_data(r.render())

    print(f"{args.loops} loops x {T} frames = {N} frames at {args.fps} fps "
          f"({N / args.fps:.1f} s), camera through {args.turns:g} revolutions")
    print(f"wrote {args.out_mp4}")


if __name__ == "__main__":
    main()
