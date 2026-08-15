#!/usr/bin/env python3
"""Does the reference demand more speed or torque than the G1 has?

    MUJOCO_GL=egl .venv/bin/python tools/dynamic_feasibility.py <csv> [...]

Static balance turned out not to explain the tracking failure — our reference
scores better on it than motions SONIC already tracks. The next cheapest
hypothesis is that the reference is simply too fast or too strong for the hardware:
a joint asked to exceed its rated speed, or a torque beyond what the motor can
produce, cannot be tracked no matter how good the policy is.

Velocities come from differencing the reference. Torques come from inverse
dynamics — given where the robot is, how fast it is moving and how hard it is
accelerating, what generalised force does that require?

The limits are taken from the URDF (effort 1-139 Nm, velocity 6.9-37 rad/s),
because SONIC's own MJCF is a kinematic model: it carries no force range, damping
or armature at all, so reading limits from it would have reported everything as
fine.

Judged against BONES-SEED motions that already track, since only a relative answer
is meaningful — inverse dynamics without contact puts the whole ground reaction in
the root residual, so absolute torque numbers flatter no one.
"""

import os
import re
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from g1_columns import joint_cols, was_reordered

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
URDF = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                    "assets", "robot_description", "urdf", "g1", "main.urdf")


def urdf_limits():
    s = open(URDF).read()
    out = {}
    for n, a in re.findall(r'<joint name="([^"]+)"[^>]*>(.*?)</joint>', s, re.S):
        e = re.search(r'effort="([\d.]+)"', a)
        v = re.search(r'velocity="([\d.]+)"', a)
        if e and v:
            out[n] = (float(e.group(1)), float(v.group(1)))
    return out


def analyse(path, model, data, lim, fps=60.0):
    df = pd.read_csv(path)
    jc = joint_cols(df)
    jn = [c[:-4] for c in jc]
    eff = np.array([lim.get(n, (100.0, 20.0))[0] for n in jn])
    vmax = np.array([lim.get(n, (100.0, 20.0))[1] for n in jn])

    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    q = np.deg2rad(df[jc].values)
    T = len(df)
    dt = 1.0 / fps

    qd = np.zeros_like(q)
    qd[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    qdd = np.zeros_like(q)
    qdd[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / dt ** 2

    tau = np.zeros((T, len(jn)))
    for i in range(T):
        data.qpos[:3] = root[i]
        data.qpos[3:7] = quat[i]
        data.qpos[7:] = q[i]
        data.qvel[:] = 0.0
        data.qvel[6:] = qd[i]
        data.qacc[:] = 0.0
        data.qacc[6:] = qdd[i]
        mujoco.mj_inverse(model, data)
        tau[i] = data.qfrc_inverse[6:]

    vpk = np.abs(qd).max(axis=0)
    tpk = np.abs(tau).max(axis=0)
    v_over = vpk / vmax
    t_over = tpk / eff
    frac_v = float((np.abs(qd) > vmax).mean())
    frac_t = float((np.abs(tau) > eff).mean())

    print(f"\n\033[1m{os.path.basename(path)}\033[0m  {T} frames")
    print(f"  peak joint speed   {100 * v_over.max():5.0f}% of limit "
          f"({jn[int(v_over.argmax())].replace('_joint', '')})")
    print(f"  peak joint torque  {100 * t_over.max():5.0f}% of limit "
          f"({jn[int(t_over.argmax())].replace('_joint', '')})")
    print(f"  \033[1mframe-joint pairs over speed limit  {100 * frac_v:5.2f}%"
          f"   over torque limit {100 * frac_t:5.2f}%\033[0m")
    worst = np.argsort(-t_over)[:3]
    print("  worst torques: " + ", ".join(
        f"{jn[k].replace('_joint', '')} {100 * t_over[k]:.0f}%" for k in worst))
    return dict(v=v_over, t=t_over, frac_v=frac_v, frac_t=frac_t)


def main(paths):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    lim = urdf_limits()
    print(f"limits from URDF: {len(lim)} joints")
    return {p: analyse(p, model, data, lim) for p in paths}


if __name__ == "__main__":
    main(sys.argv[1:])
