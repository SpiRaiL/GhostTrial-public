#!/usr/bin/env python3
"""Retarget an edited SOMA BVH onto the G1, so human edits reach the robot.

    vendor/GEM-X/.venv/bin/python tools/retarget_bvh_to_g1.py <in.bvh> <out.csv>

GEM-X's demo only ever retargets straight from its own solve, so once the human
motion is edited by hand there is no path to the robot. This is that path: it is
the same NewtonPipeline call the demo's BVH round-trip uses, pointed at a file of
ours instead of one it just wrote.

Editing the human and re-retargeting is the right way round. Adjusting the G1 CSV
directly would mean re-deriving the same change twice, in two joint conventions,
and the .blend and the training data would drift apart the first time one was
touched without the other.

Runs under GEM-X's own venv, and chdir's into the repo because the retargeter
resolves its robot configs relative to the repo root.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMX = os.path.join(REPO, "vendor", "GEM-X")
SRC = os.path.abspath(sys.argv[1])
DST = os.path.abspath(sys.argv[2])

sys.path.insert(0, os.path.join(GEMX, "scripts", "demo"))
os.chdir(GEMX)

import warp as wp  # noqa: E402
from soma_retargeter.assets.bvh import load_bvh  # noqa: E402
from soma_retargeter.assets.csv import save_csv  # noqa: E402
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline  # noqa: E402
from soma_retargeter.utils.space_conversion_utils import (  # noqa: E402
    FacingDirectionType, SpaceConverter)

bvh_skeleton, anim = load_bvh(SRC)
space = SpaceConverter(FacingDirectionType.MUJOCO)
offset = space.transform(wp.transform_identity())

pipeline = NewtonPipeline(bvh_skeleton, "soma", "unitree_g1")
pipeline.add_input_motions([anim], [offset], scale_animation=True)
out = pipeline.execute()

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_csv(DST, out[0])
print(f"retargeted {os.path.basename(SRC)} -> {DST}")
