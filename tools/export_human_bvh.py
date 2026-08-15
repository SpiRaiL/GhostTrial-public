#!/usr/bin/env python3
"""Export the HUMAN motion GEM-X reconstructed, before any robot adaptation.

    vendor/GEM-X/.venv/bin/python tools/export_human_bvh.py

GEM-X's demo writes the G1 retarget but throws the intermediate SOMA skeleton away.
This pulls `pred_body_params_global` out of hpe_results.pt, rebuilds the performer's
own SOMA skeleton from the identity/scale coefficients the model solved for, and
writes it as BVH — the 78-joint human, at his real proportions, in world space.

That is the thing worth checking before we adapt anything: if the human
reconstruction is wrong, no amount of retargeting will save it. It is also the
artefact with value outside this project, since it is not G1-specific.
"""

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMX = os.path.join(REPO, "vendor", "GEM-X")
sys.path.insert(0, os.path.join(GEMX, "scripts", "demo"))
os.chdir(GEMX)                      # retarget_utils resolves config files relative to the repo

from retarget_utils import build_soma_skeleton_from_model, export_soma_bvh  # noqa: E402

OUT = os.path.join(REPO, "data", "human_bvh")
os.makedirs(OUT, exist_ok=True)

for take in sorted(os.listdir(os.path.join(REPO, "data", "gemx_output"))):
    pt = os.path.join(REPO, "data", "gemx_output", take, "hpe_results.pt")
    if not os.path.exists(pt):
        continue
    d = torch.load(pt, map_location="cpu", weights_only=False)
    # top-level body_params_global is already (T, D) — pass it through exactly as
    # run_retarget() does, so the skeleton is built the same way the G1 path built it
    params = d["body_params_global"]
    skel = build_soma_skeleton_from_model(params["identity_coeffs"], params["scale_params"])

    dst = os.path.join(OUT, f"{take}_human.bvh")
    export_soma_bvh(params, skel, 60.0, dst)
    n = params["body_pose"].shape[0]
    print(f"{take}: {n} frames, {skel.num_joints} joints -> {dst}")
