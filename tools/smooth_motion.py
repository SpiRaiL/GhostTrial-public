#!/usr/bin/env python3
"""Take the jitter out of a retargeted motion.

    MUJOCO_GL=egl .venv/bin/python tools/smooth_motion.py in.csv out.csv --cutoff 4

Opening the takes in Blender's graph editor shows visibly noisy curves, and the
numbers agree: our joint accelerations average 19.8 rad/s^2 against 0.9-8.2 for the
BONES-SEED motions SONIC already tracks. That is 2-20x, and it is a tracking
policy's problem directly — every spike is a target the policy is asked to chase
and cannot, so it spends its control authority on noise instead of on the move.

Notably this is NOT high-frequency hiss: our energy above 6 Hz is lower than
several of the seed clips. It is large accelerations at moderate frequency, which
is what a per-frame IK solve produces when consecutive frames each find a slightly
different solution to the same underdetermined problem.

Zero-phase Butterworth, so the motion does not shift in time — a one-way filter
would lag the whole clip and quietly break the alignment with the reference.
The root translation and rotation are filtered too, since the same solve produces
them.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

ap = argparse.ArgumentParser()
ap.add_argument("in_csv")
ap.add_argument("out_csv")
ap.add_argument("--cutoff", type=float, default=4.0, help="low-pass corner, Hz")
ap.add_argument("--fps", type=float, default=60.0)
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
jc = [c for c in df.columns if c.endswith("_dof")]
tc = ["root_translateX", "root_translateY", "root_translateZ"]
rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]

b, a = butter(2, args.cutoff / (args.fps / 2), btype="low")
out = df.copy()


def acc_of(v):
    return float(np.abs(np.diff(np.deg2rad(v), n=2, axis=0) * args.fps ** 2).mean())


before = acc_of(df[jc].values)

# joints and root translation filter directly
for col in jc + tc:
    out[col] = filtfilt(b, a, df[col].values)

# root rotation goes through quaternions — filtering Euler angles independently
# tears the orientation apart wherever one of them wraps
R = Rotation.from_euler("xyz", df[rc].values, degrees=True)
q = R.as_quat()
q = q * np.sign(np.einsum("ij,ij->i", q, np.roll(q, 1, axis=0)))[:, None]  # unwrap
qf = np.stack([filtfilt(b, a, q[:, k]) for k in range(4)], axis=1)
qf /= np.linalg.norm(qf, axis=1, keepdims=True)
out[rc] = Rotation.from_quat(qf).as_euler("xyz", degrees=True)

after = acc_of(out[jc].values)
dev = np.abs(out[jc].values - df[jc].values)
print(f"{os.path.basename(args.in_csv)}  {len(df)} frames, cutoff {args.cutoff:g} Hz")
print(f"  mean |joint accel|   {before:6.1f} -> {after:6.1f} rad/s^2"
      f"   (BONES-SEED range 0.9-8.2)")
print(f"  pose shifted by      median {np.median(dev):.2f} deg, p99 {np.percentile(dev, 99):.2f} deg")
out.to_csv(args.out_csv, index=False, float_format="%.6f")
print(f"wrote {args.out_csv}")
