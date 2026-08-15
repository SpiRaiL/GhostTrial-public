#!/usr/bin/env python3
"""Mirror a BONES-SEED-format G1 CSV through the robot's sagittal plane.

    .venv/bin/python tools/mirror_csv.py in.csv out.csv

Reuses the joint convention validated against BONES-SEED's own `_M` pairs
(mean 0.95 deg, max 5.5 deg -- the residual is independent-IK noise, since their
mirrors are retargeted separately rather than reflected). The root is reflected
in the clip's frame-0 heading frame, not about world zero.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bvh_to_bones_csv import JOINT_NAMES, mirror  # noqa: E402

ROOT_T = ["root_translateX", "root_translateY", "root_translateZ"]
ROOT_R = ["root_rotateX", "root_rotateY", "root_rotateZ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    dof_cols = [f"{n}_dof" for n in JOINT_NAMES]
    missing = [c for c in dof_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing dof columns: {missing[:3]}...")

    rc, rd, dd = mirror(df[ROOT_T].values, df[ROOT_R].values, df[dof_cols].values)
    out = df.copy()
    out[ROOT_T] = rc
    out[ROOT_R] = rd
    out[dof_cols] = dd

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format="%.8f")
    print(f"mirrored {len(out)} frames -> {args.out_csv}")


if __name__ == "__main__":
    main()
