# Magic attack (hadouken) — working pipeline

End-to-end, verified 2026-08-07. Every step below runs locally on the laptop,
no GPU. Only the fine-tune needs Nebius.

## Commands

```bash
cd ~/RC/competitions/GhostTrial

# 1. FBX -> BVH, trimmed to the strike, in CENTIMETRES (GMR's loader wants cm)
/snap/bin/blender --background --python tools/fbx_to_bvh.py -- \
    raw/mixamo/standing_2h_magic_attack_04.fbx \
    raw/mixamo/bvh/magic_attack_trim.bvh 18 52 1.0

# 2. Retarget to G1 29-DoF and emit a BONES-SEED-format CSV (+ mirrored copy)
.venv/bin/python tools/bvh_to_bones_csv.py \
    raw/mixamo/bvh/magic_attack_trim.bvh \
    data/mixamo_csv/magic_attack/magic_attack.csv --mirror

# 3. SONIC's own converter -> motion_lib PKL
.venv/bin/python \
    vendor/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input data/mixamo_csv/magic_attack \
    --output data/motion_lib_magic/robot \
    --fps 30 --fps_source 30 --individual
```

Output: `data/motion_lib_magic/robot/magic_attack/{magic_attack,magic_attack_M}.pkl`
— `root_trans_offset (35,3)`, `pose_aa (35,30,3)`, `dof (35,29)`,
`root_rot (35,4)`, `smpl_joints (35,24,3)`, `fps=30`.

## Why these choices

**`--format nokov`, not `lafan1`.** GMR's BVH loader derives foot orientation
from `LeftToe`/`RightToe` under `lafan1` but from `LeftToeBase`/`RightToeBase`
under `nokov` — and `LeftToeBase` is what Mixamo calls it. Every other joint
`bvh_nokov_to_g1.json` needs (`Hips`, `Spine2`, `Left/RightUpLeg`,
`Left/RightLeg`, `Left/RightArm`, `Left/RightForeArm`, `Left/RightHand`) is
already present in the Mixamo rig once `mixamorig:` is stripped. So a Mixamo
clip retargets with **no config changes at all**.

**Centimetres.** `general_motion_retargeting/utils/lafan1.py` divides BVH
positions by 100. A metres BVH makes the retargeter think the performer is
1.75 cm tall.

**No DOF permutation.** GMR's `g1_mocap_29dof.xml` emits its 29 actuators in
exactly the BONES-SEED CSV column order. Verified name by name. Only unit
conversion (rad→deg, m→cm) is applied.

**`--fps_source 30`.** BONES-SEED is 120 fps so the docs use `--fps_source 120`;
Mixamo is 30 fps and already at the target rate, so no decimation.

## Validation performed

- **Schema** — column names byte-identical to a real BONES-SEED CSV.
- **Ranges** — root height, root rotation and per-joint values all sit inside
  the envelope of `shadow_boxing_R_001__A359`.
- **Joint limits** — all 29 DOFs within the G1's `jnt_range`, **0 violations**.
- **Mirror (joints)** — checked against BONES-SEED's real `_M` pair for
  `shadow_boxing_R_001__A359`: mean 0.95°, max 5.5° error. The residual is
  independent-IK noise, not a convention error — BONES-SEED's mirrors are
  retargeted separately from mirrored human BVH, they are not algebraic
  mirrors of the G1 CSV. Root Z matched to 0.02 cm, confirming alignment.
- **Mirror (root)** — *cannot* be validated that way, since the reference root
  trajectory is an independent take (X/Y differ by ~the size of the travel
  itself). The implementation reflects in the robot's frame-0 heading frame,
  which is the correct convention; unverified against ground truth.

## Known risks going into training

1. **Several joints saturate at their limits** during the retarget:
   `left_elbow` (−60.0), `left_wrist_roll` (−113.0), `right_ankle_pitch`
   (−50.0), `right_ankle_roll` (+15.0), `waist_pitch` (+29.8). The human pose
   exceeds what the G1 can reach and gets clipped, so the robot's version of
   the thrust is a compromise. Both ankles pinned at their limits is the
   worrying one — that is the balance margin gone.
2. **The 19 cm hip drop** noted in `raw/mixamo/README.md` drives straight into
   that ankle saturation.
3. **35 frames / 1.17 s from a single clip.** Mirroring gives 2. That is a very
   small fine-tuning set; expect to need Kimodo variants or more Mixamo
   spell/cast clips for the policy to generalise past one trajectory.

Next: preview in MuJoCo before spending GPU time. Watching the retarget play
back is much cheaper than discovering the ankle clipping after a training run.
