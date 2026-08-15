"""Put a G1 motion's joint columns in MuJoCo order, by name.

Every tool here loads a CSV and assigns it straight into MuJoCo:

    data.qpos[7:] = np.deg2rad(df[joint_cols].values)

which is only correct if joint_cols is in MuJoCo's own joint order. BONES-SEED
CSVs are. **Policy rollouts are not** — tools/rollout_to_csv.py names its columns
from IsaacLab's robot.data.joint_names, which interleaves the two legs and the
waist:

    MuJoCo    left_hip_pitch, left_hip_roll,  left_hip_yaw,  left_knee, ...
    IsaacLab  left_hip_pitch, right_hip_pitch, waist_yaw,    left_hip_roll, ...

Same 29 joints, different order. So every rollout was analysed and rendered with
its joints permuted: the videos showed a robot that was not the one the policy
produced, and the foot-contact, stance-width and balance numbers taken from them
were meaningless. Two different policies scoring identical foot heights to 0.1 cm
was the tell.

The columns carry their own names, so the fix is to reorder by name rather than
trust position. Use this everywhere instead of a bare `endswith("_dof")` filter.
"""

import os

import mujoco

_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                    "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
_ORDER = None


def mujoco_joint_names():
    global _ORDER
    if _ORDER is None:
        m = mujoco.MjModel.from_xml_path(_XML)
        _ORDER = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                  for j in range(1, m.njnt)]
    return _ORDER


def joint_cols(df, strict=True):
    """The df's *_dof columns, ordered to match MuJoCo's qpos layout."""
    have = {c[:-4]: c for c in df.columns if c.endswith("_dof")}
    cols, missing = [], []
    for n in mujoco_joint_names():
        if n in have:
            cols.append(have[n])
        else:
            missing.append(n)
    if strict and missing:
        raise SystemExit(f"columns missing for MuJoCo joints: {missing}")
    extra = set(have) - set(mujoco_joint_names())
    if strict and extra:
        raise SystemExit(f"columns not present in the MuJoCo model: {sorted(extra)}")
    return cols


def was_reordered(df):
    """True if the file's own column order differs from MuJoCo's."""
    raw = [c for c in df.columns if c.endswith("_dof")]
    return raw != joint_cols(df)
