#!/usr/bin/env python3
"""Author G1 motion directly in the robot's own 29-DoF joint space.

    MUJOCO_GL=egl .venv/bin/python tools/g1_author.py --out data/g1_authored/combo
    MUJOCO_GL=egl .venv/bin/python tools/g1_author.py --probe stance --png /tmp/p.png

Why not author on a human rig and retarget: that is what `author_spear_throw.py`
does, and it inherits exactly the human->G1 mismatch that pins the Mixamo clip's
ankles. Poses that are comfortable for a person bury the G1's bulky shoulder and
thigh links inside the torso — 805 self-collision contacts against the Mixamo
baseline's 5. Authoring in joint space instead gives three things for free:

  * limit violations are impossible — every value is clamped to `jnt_range`
  * self-collision is measurable per pose, so a bad pose is caught at authoring time
  * there is no retarget step at all; 29 DoF + root IS the BONES-SEED CSV format

Conventions, measured (not assumed) from the MJCF — see the calibration in the
session log:
    +X forward (the toes point +X), +Y left, +Z up
    hip_pitch    negative = knee forward        knee   positive = flex
    shoulder_pitch negative = arm rotates UP; elbow positive = flex.
    NOTE the zero pose has the arms straight FORWARD and horizontal, not hanging
    down - so shoulder_pitch 0 + elbow 0 is already a forward reach, and -90 puts
    the arm vertically overhead. Swept and verified, do not assume otherwise.
    shoulder_roll  positive = toward the robot's LEFT, so + abducts the left arm
                   and adducts the right; right arm out is negative
    waist_yaw    positive = turn left
"""

import argparse
import os

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
IDX = {n: i for i, n in enumerate(JOINTS)}


def model():
    from general_motion_retargeting import params
    return mujoco.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"]))


# ── the poses ─────────────────────────────────────────────────────────────────
# Bladed southpaw-ish stance: left foot forward, right arm throws. Values are
# degrees. Anything omitted is 0.

STANCE = dict(
    left_hip_pitch=-24, left_knee=32, left_ankle_pitch=-12, left_hip_roll=6,
    right_hip_pitch=6, right_knee=26, right_ankle_pitch=-16, right_hip_roll=-6,
    waist_yaw=-16, waist_pitch=8,
    left_shoulder_pitch=-18, left_shoulder_roll=20, left_elbow=82,
    right_shoulder_pitch=-14, right_shoulder_roll=-22, right_elbow=92,
)

COIL = dict(
    STANCE,
    left_hip_pitch=-18, left_knee=26,
    right_hip_pitch=12, right_knee=36, right_ankle_pitch=-20,
    waist_yaw=-36, waist_pitch=6,
    left_shoulder_pitch=-24, left_shoulder_roll=26, left_elbow=90,
    right_shoulder_pitch=-34, right_shoulder_roll=-44, right_shoulder_yaw=-30,
    right_elbow=95,
)

THROW = dict(
    STANCE,
    left_hip_pitch=-34, left_knee=40, left_ankle_pitch=-16,
    right_hip_pitch=16, right_knee=16, right_ankle_pitch=-22,
    waist_yaw=22, waist_pitch=14,
    left_shoulder_pitch=-16, left_shoulder_roll=30, left_elbow=94,
    right_shoulder_pitch=-2, right_shoulder_roll=-8, right_shoulder_yaw=4,
    right_elbow=2, right_wrist_pitch=-8,
)

# uppercut: sink into a deep crouch, then drive up with the right arm rising
CROUCH = dict(
    STANCE,
    left_hip_pitch=-78, left_knee=124, left_ankle_pitch=-40, left_hip_roll=11,
    right_hip_pitch=-66, right_knee=118, right_ankle_pitch=-38, right_hip_roll=-11,
    waist_yaw=-8, waist_pitch=26,
    left_shoulder_pitch=-20, left_shoulder_roll=24, left_elbow=94,
    right_shoulder_pitch=-6, right_shoulder_roll=-24, right_elbow=95,
)

RISE = dict(
    STANCE,
    left_hip_pitch=-12, left_knee=10, left_ankle_pitch=-4,
    right_hip_pitch=2, right_knee=8, right_ankle_pitch=-8,
    waist_yaw=12, waist_pitch=-10,
    left_shoulder_pitch=-14, left_shoulder_roll=26, left_elbow=80,
    right_shoulder_pitch=-138, right_shoulder_roll=-12, right_shoulder_yaw=8,
    right_elbow=14,
)

# frame, pose, root xyz (m, z is a nominal pelvis height — grounding fixes it), yaw
KEYS = [
    (0,   STANCE, (0.00, 0.0, 0.74), 0),
    (14,  STANCE, (0.00, 0.0, 0.74), 0),
    (40,  COIL,   (-0.03, 0.0, 0.73), 0),
    (52,  THROW,  (0.09, 0.0, 0.72), 0),
    (66,  THROW,  (0.09, 0.0, 0.72), 0),
    (80,  STANCE, (0.04, 0.0, 0.74), 0),
    (104, CROUCH, (0.02, 0.0, 0.40), 0),
    (120, CROUCH, (0.02, 0.0, 0.40), 0),
    (138, RISE,   (0.12, 0.0, 0.86), 0),
    (150, RISE,   (0.12, 0.0, 0.86), 0),
    (172, STANCE, (0.08, 0.0, 0.74), 0),
]

PROBES = {"stance": STANCE, "coil": COIL, "throw": THROW, "crouch": CROUCH, "rise": RISE}


def to_vec(pose):
    v = np.zeros(len(JOINTS))
    for k, d in pose.items():
        if k not in IDX:
            raise KeyError(f"unknown joint {k!r}")
        v[IDX[k]] = d
    return v


def clamp(m, deg):
    """Clamp to the MJCF's own jnt_range. Makes limit violations impossible."""
    out = deg.copy()
    for i, n in enumerate(JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")
        lo, hi = np.rad2deg(m.jnt_range[jid])
        out[i] = np.clip(out[i], lo, hi)
    return out


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def build(m):
    n = KEYS[-1][0] + 1
    dof = np.zeros((n, len(JOINTS)))
    root = np.zeros((n, 3))
    yaw = np.zeros(n)
    for (f0, p0, r0, y0), (f1, p1, r1, y1) in zip(KEYS, KEYS[1:]):
        a, b = to_vec(p0), to_vec(p1)
        for f in range(f0, f1 + 1):
            t = smoothstep((f - f0) / max(f1 - f0, 1))
            dof[f] = a + (b - a) * t
            root[f] = np.array(r0) + (np.array(r1) - np.array(r0)) * t
            yaw[f] = y0 + (y1 - y0) * t
    for f in range(n):
        dof[f] = clamp(m, dof[f])
    return dof, root, yaw


def self_collisions(m, d, dof, root, yaw):
    floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    total, worst, pairs = 0, 0.0, {}
    for i in range(len(dof)):
        d.qpos[:3] = root[i]
        d.qpos[3:7] = Rotation.from_euler("z", yaw[i], degrees=True).as_quat()[[3, 0, 1, 2]]
        d.qpos[7:] = np.deg2rad(dof[i])
        mujoco.mj_forward(m, d)
        for k in range(d.ncon):
            g1, g2 = d.contact.geom1[k], d.contact.geom2[k]
            if g1 in floor or g2 in floor:
                continue
            total += 1
            worst = min(worst, d.contact.dist[k])
            nm = tuple(sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
                              for g in (g1, g2)))
            pairs[nm] = pairs.get(nm, 0) + 1
    return total, worst, sorted(pairs.items(), key=lambda kv: -kv[1])[:6]


def ground(m, d, dof, root, yaw):
    """Drop every frame so the lowest geom touches z=0 (no flight phase here)."""
    floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    out = root.copy()
    for i in range(len(dof)):
        lo, hi = -0.5, 0.5
        for _ in range(44):
            mid = 0.5 * (lo + hi)
            d.qpos[:3] = root[i] + [0, 0, mid]
            d.qpos[3:7] = Rotation.from_euler("z", yaw[i], degrees=True).as_quat()[[3, 0, 1, 2]]
            d.qpos[7:] = np.deg2rad(dof[i])
            mujoco.mj_forward(m, d)
            g = [d.contact.dist[k] for k in range(d.ncon)
                 if d.contact.geom1[k] in floor or d.contact.geom2[k] in floor]
            if (min(g) if g else 1.0) > 0.0:
                hi = mid
            else:
                lo = mid
        out[i, 2] += hi
    return out


def write_csv(path, dof, root, yaw):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hdr = (["Frame", "root_translateX", "root_translateY", "root_translateZ",
            "root_rotateX", "root_rotateY", "root_rotateZ"] + [f"{n}_joint_dof" for n in JOINTS])
    rows = np.column_stack([np.arange(len(dof)), root * 100.0,
                            np.zeros(len(dof)), np.zeros(len(dof)), yaw, dof])
    np.savetxt(path, rows, delimiter=",", header=",".join(hdr), comments="",
               fmt=["%d"] + ["%.8f"] * (rows.shape[1] - 1))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/g1_authored/combo")
    ap.add_argument("--probe", choices=list(PROBES))
    ap.add_argument("--png", default="/tmp/g1_probe.png")
    args = ap.parse_args()

    m = model()
    d = mujoco.MjData(m)

    if args.probe:
        from PIL import Image
        dof = clamp(m, to_vec(PROBES[args.probe]))[None, :]
        root = ground(m, d, dof, np.array([[0.0, 0.0, 0.74]]), np.zeros(1))
        r = mujoco.Renderer(m, height=640, width=520)
        tiles = []
        for az in (0, 45, 90):
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = az, -8, 2.6
            cam.lookat[:] = [root[0, 0], 0, 0.7]
            d.qpos[:3] = root[0]
            d.qpos[3:7] = [1, 0, 0, 0]
            d.qpos[7:] = np.deg2rad(dof[0])
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            tiles.append(Image.fromarray(r.render()))
        sheet = Image.new("RGB", (520 * 3, 640))
        for k, t in enumerate(tiles):
            sheet.paste(t, (520 * k, 0))
        sheet.save(args.png)
        n, worst, pairs = self_collisions(m, d, dof, root, np.zeros(1))
        print(f"pose {args.probe}: self-collisions {n}, deepest {worst*1000:.1f} mm")
        for p, c in pairs:
            print(f"   {c:3d}  {p[0]} <-> {p[1]}")
        print(f"wrote {args.png}")
        return

    dof, root, yaw = build(m)
    root = ground(m, d, dof, root, yaw)
    n, worst, pairs = self_collisions(m, d, dof, root, yaw)
    print(f"{len(dof)} frames @ 30 fps ({len(dof)/30:.2f} s)")
    print(f"joint limits: 0 violations by construction (clamped to jnt_range)")
    print(f"self-collision contacts: {n}   deepest {worst*1000:.1f} mm")
    for p, c in pairs:
        print(f"   {c:4d}  {p[0]} <-> {p[1]}")
    print(f"pelvis height {root[:,2].min()*100:.1f}..{root[:,2].max()*100:.1f} cm")
    p = write_csv(f"{args.out}/spear_uppercut.csv", dof, root, yaw)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
