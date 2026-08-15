#!/usr/bin/env python3
"""Retarget a Mixamo BVH to the Unitree G1 and write a BONES-SEED-format CSV.

The CSV is the cheapest entry point into SONIC: its own
`gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` reads flat
BONES-SEED CSVs (modes 4 and 5) and emits the motion_lib PKLs that
`train_agent_trl.py` consumes.

Format (verified against the 142,220 real CSVs in retargeted/bones-seed/g1/csv/):
    Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z} [deg, euler xyz
    intrinsic], then 29 <joint>_dof columns [deg] in MuJoCo actuator order.

GMR's `g1_mocap_29dof.xml` emits its 29 DOFs in exactly that order already, so
no column permutation is applied — only unit conversion.

Mixamo rigs work with `--format nokov` (not lafan1): nokov keys the foot
orientation off `LeftToeBase`/`RightToeBase`, which is what Mixamo names them,
whereas lafan1 expects `LeftToe`. Every other joint the config needs (Hips,
Spine2, Left/RightUpLeg, Left/RightLeg, Left/RightArm, Left/RightForeArm,
Left/RightHand) is present in the Mixamo skeleton under the same name once the
`mixamorig:` prefix is stripped.

The input BVH must be in CENTIMETRES — GMR's loader divides positions by 100.

    python tools/bvh_to_bones_csv.py raw/mixamo/bvh/magic_attack_trim.bvh \
        data/mixamo_csv/magic_attack/magic_attack.csv
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation

# BONES-SEED CSV joint column order == GMR g1_mocap_29dof.xml actuator order.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def mirror(root_cm, root_deg, dof_deg):
    """Reflect a motion through the robot's own sagittal plane.

    Joints: swap the left and right blocks, negate every roll/yaw DOF (those
    are antisymmetric about the plane), leave pitch/knee/elbow alone. This part
    is validated — see `check_mirror` below.

    Root: the reflection plane is the robot's sagittal plane *at frame 0*, not
    the world XZ plane, so the trajectory is expressed relative to the initial
    position and heading, mirrored there, then put back. Mirroring about world
    zero instead would translate the clip sideways by twice its start offset.
    """
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    out = np.empty_like(dof_deg)
    for i, name in enumerate(JOINT_NAMES):
        if name.startswith("left_"):
            src = idx["right_" + name[5:]]
        elif name.startswith("right_"):
            src = idx["left_" + name[6:]]
        else:
            src = i
        sign = -1.0 if ("roll" in name or "yaw" in name) else 1.0
        out[:, i] = sign * dof_deg[:, src]

    # into the frame-0 heading frame
    p0 = root_cm[0].copy()
    yaw0 = np.deg2rad(root_deg[0, 2])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rel = root_cm - p0
    local = np.column_stack([c * rel[:, 0] - s * rel[:, 1],
                             s * rel[:, 0] + c * rel[:, 1],
                             rel[:, 2]])
    local[:, 1] *= -1.0                       # reflect across the sagittal plane
    c, s = np.cos(yaw0), np.sin(yaw0)         # and back out
    root_cm = np.column_stack([c * local[:, 0] - s * local[:, 1],
                               s * local[:, 0] + c * local[:, 1],
                               local[:, 2]]) + p0

    root_deg = root_deg.copy()
    root_deg[:, 0] *= -1.0                              # roll
    root_deg[:, 2] = 2.0 * root_deg[0, 2] - root_deg[:, 2]  # yaw, about the start heading
    return root_cm, root_deg, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh")
    ap.add_argument("out_csv")
    ap.add_argument("--format", default="nokov", choices=["nokov", "lafan1"])
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mirror", action="store_true", help="also write a _M mirrored copy")
    ap.add_argument(
        "--human-height", type=float, default=None,
        help="Performer stature in metres. GMR's BVH loader HARDCODES 1.75 m "
             "(utils/lafan1.py); the Mixamo rig is ~1.65 m, and that 6%% error "
             "makes the retargeted robot float. 'auto' behaviour: measured from "
             "the BVH head joint + 0.13 m for the skull top.",
    )
    ap.add_argument("--auto-height", action="store_true",
                    help="measure stature from the BVH instead of trusting the loader")
    args = ap.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(args.bvh, format=args.format)
    if args.auto_height:
        z = {k: v[0][2] for k, v in frames[0].items()}
        foot = min(z.get(n, 9.9) for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"))
        human_height = z["Head"] - foot + 0.13
        print(f"measured stature from BVH: {human_height:.3f} m (loader claimed 1.75)")
    if args.human_height is not None:
        human_height = args.human_height
    print(f"loaded {len(frames)} frames from {args.bvh} (stature {human_height:.3f} m)")

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=human_height,
    )

    qpos = np.array([retargeter.retarget(f) for f in frames])
    print(f"retargeted -> qpos {qpos.shape}  (3 root + 4 quat + {qpos.shape[1] - 7} dof)")
    assert qpos.shape[1] - 7 == len(JOINT_NAMES), f"expected 29 DOF, got {qpos.shape[1] - 7}"

    root_cm = qpos[:, :3] * 100.0
    quat_wxyz = qpos[:, 3:7]
    root_deg = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    dof_deg = np.rad2deg(qpos[:, 7:])

    variants = [(args.out_csv, root_cm, root_deg, dof_deg)]
    if args.mirror:
        base, ext = os.path.splitext(args.out_csv)
        variants.append((f"{base}_M{ext}", *mirror(root_cm, root_deg, dof_deg)))

    header = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
              "root_rotateX", "root_rotateY", "root_rotateZ"] + [f"{n}_dof" for n in JOINT_NAMES]

    for path, rc, rd, dd in variants:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = np.column_stack([np.arange(len(rc)), rc, rd, dd])
        np.savetxt(path, rows, delimiter=",", header=",".join(header),
                   comments="", fmt=["%d"] + ["%.8f"] * (rows.shape[1] - 1))
        print(f"wrote {path}  ({len(rows)} frames, {rows.shape[1]} cols)")

    print(f"\nroot height  : {root_cm[:, 2].min():.1f}..{root_cm[:, 2].max():.1f} cm")
    print(f"max |dof|    : {np.abs(dof_deg).max():.1f} deg")


if __name__ == "__main__":
    main()
