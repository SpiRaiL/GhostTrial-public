#!/usr/bin/env python3
"""Render every clip/still the progress report needs, offscreen on the local GPU.

    MUJOCO_GL=egl .venv/bin/python tools/make_report_media.py "<report dir>"
"""

import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

OUT = sys.argv[1]
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

from general_motion_retargeting import params  # noqa: E402

MODEL = mujoco.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"]))
DATA = mujoco.MjData(MODEL)
W, H = 900, 700


def load(csv):
    df = pd.read_csv(csv)
    jc = [c for c in df.columns if c.endswith("_dof")]
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values, degrees=True
    ).as_quat()[:, [3, 0, 1, 2]]
    return root, quat, np.deg2rad(df[jc].values)


def frames(csv, azimuth, elevation=-10.0, distance=2.7, w=W, h=H, lookat_z=0.75):
    root, quat, dof = load(csv)
    r = mujoco.Renderer(MODEL, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = [root[:, 0].mean(), root[:, 1].mean(), lookat_z]
    out = []
    for i in range(len(root)):
        DATA.qpos[:3] = root[i]
        DATA.qpos[3:7] = quat[i]
        DATA.qpos[7:] = dof[i]
        mujoco.mj_forward(MODEL, DATA)
        r.update_scene(DATA, camera=cam)
        out.append(Image.fromarray(r.render()))
    return out


def write_mp4(imgs, path, fps=30, slowdown=3):
    tmp = tempfile.mkdtemp(prefix="rep_")
    n = 0
    for im in imgs:
        for _ in range(slowdown):
            im.save(os.path.join(tmp, f"{n:05d}.png"))
            n += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", path], check=True)
    print("wrote", path)


G = f"{REPO}/data/mixamo_csv/magic_attack/magic_attack.csv"
RAW = f"{REPO}/data/mixamo_csv/magic_attack_raw/magic_attack.csv"

# 1 · three-quarter and side views of the final, grounded clip
write_mp4(frames(G, 45), f"{OUT}/magic_attack_3q.mp4")
write_mp4(frames(G, 0), f"{OUT}/magic_attack_side.mp4")

# 2 · before/after grounding, side by side with a floor rule
pre, post = frames(RAW, 0, distance=2.2, lookat_z=0.55), frames(G, 0, distance=2.2, lookat_z=0.55)
pair = []
for a, b in zip(pre, post):
    c = Image.new("RGB", (a.width + b.width + 8, a.height), (16, 18, 22))
    c.paste(a, (0, 0))
    c.paste(b, (a.width + 8, 0))
    d = ImageDraw.Draw(c)
    d.text((14, 12), "BEFORE — floats 19–57 mm", fill=(224, 115, 107), font=FONT_S)
    d.text((a.width + 22, 12), "AFTER — grounded, 0.0 mm", fill=(78, 201, 163), font=FONT_S)
    pair.append(c)
write_mp4(pair, f"{OUT}/grounding_before_after.mp4")

# 3 · key-frame montage: charge / peak speed / full extension
KEY = [(5, "charge  f23"), (11, "peak speed  f29  5.84 m/s"), (23, "extension  f41")]
tiles = frames(G, 45, distance=2.5)
sheet = Image.new("RGB", (W * len(KEY), H), (16, 18, 22))
for k, (idx, label) in enumerate(KEY):
    sheet.paste(tiles[idx], (W * k, 0))
    ImageDraw.Draw(sheet).text((W * k + 18, 16), label, fill=(20, 24, 30), font=FONT)
sheet.save(f"{OUT}/keyframes.png")
print("wrote", f"{OUT}/keyframes.png")

# 4 · a single hero still
tiles[23].save(f"{OUT}/hero_extension.png")
print("wrote", f"{OUT}/hero_extension.png")
