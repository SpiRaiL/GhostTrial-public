#!/usr/bin/env python3
"""Render a BONES-SEED-format G1 CSV to MP4 with MuJoCo, offscreen on the GPU.

    MUJOCO_GL=egl .venv/bin/python tools/render_motion.py \
        data/mixamo_csv/magic_attack/magic_attack.csv out.mp4 --overlay

Kinematic playback: the CSV's joint angles are written straight into qpos and
the scene is rendered. No physics, no controller — this shows what the retarget
produced, which is exactly what we want to inspect before spending GPU hours
training a policy to track it.
"""

import argparse
import os
import subprocess
import tempfile

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from g1_columns import joint_cols as g1_joint_cols

JOINT_SUFFIX = "_dof"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out_mp4")
    ap.add_argument("--fps", type=int, default=60)   # the CSVs are 60 fps; rendering at 30 played every review video at HALF speed
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=315.0)   # front-ish: 135 looked at the robot's back, which hides the arms
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--slowdown", type=int, default=1, help="repeat each frame N times")
    args = ap.parse_args()

    from general_motion_retargeting import params

    xml = str(params.ROBOT_XML_DICT["unitree_g1"])
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    df = pd.read_csv(args.csv)
    # by NAME, not position: rollout CSVs are written in IsaacLab joint order
    # while qpos wants MuJoCo order, and loading one as the other renders a
    # robot the policy never produced. See tools/g1_columns.py.
    joint_cols = g1_joint_cols(df)
    assert len(joint_cols) == 29, f"expected 29 dof columns, got {len(joint_cols)}"

    root_m = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat_xyzw = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values, degrees=True
    ).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    dof_rad = np.deg2rad(df[joint_cols].values)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = [root_m[:, 0].mean(), root_m[:, 1].mean(), 0.8]

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    tmp = tempfile.mkdtemp(prefix="g1render_")
    n = 0
    for i in range(len(df)):
        data.qpos[:3] = root_m[i]
        data.qpos[3:7] = quat_wxyz[i]
        data.qpos[7:] = dof_rad[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        px = renderer.render()
        from PIL import Image
        img = Image.fromarray(px)
        for _ in range(args.slowdown):
            img.save(os.path.join(tmp, f"{n:05d}.png"))
            n += 1

    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", os.path.join(tmp, "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out_mp4],
        check=True,
    )
    print(f"wrote {args.out_mp4}  ({n} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
