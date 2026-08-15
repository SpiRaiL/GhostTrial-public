#!/usr/bin/env bash
# Export a trained SONIC checkpoint to ONNX — submission slot 1.
#
#   tools/export_onnx.sh [checkpoint.pt] [motion_dir]
#
# Writes <experiment_dir>/exported/model_step_NNNNNN_{g1,smpl,teleop,encoder}.onnx.
# The G1 one is ours: `_g1` is the encoder trained on G1-space motion, which is
# what our capture pipeline produces.
#
# num_envs MUST be 1 — eval_agent_trl.py asserts on it before exporting, and the
# assert fires only after Isaac Lab has finished starting, so getting it wrong
# costs several minutes rather than seconds.
#
# Do NOT pass +exp= here, even though training needs it. eval_agent_trl.py loads
# config.yaml from beside the checkpoint and merges the CLI over it, so adding the
# experiment again duplicates its defaults and Hydra dies with
# "manager_env/recorders appears more than once in the final defaults list".
#
# Keys the trainer takes bare (num_envs, headless) need a leading + here, because
# CLI overrides are parsed against base_eval, which does not declare them.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-$(ls -t logs_rl/TRL_G1_Track/manager/universal_token/all_modes/*/last.pt | head -1)}"
MOTION="${2:-/gt/data/motion_lib_capture/robot/a3}"
CKPT_IN_C="/opt/gr00t/${CKPT#./}"

echo "exporting from: $CKPT"
exec docker run --rm --gpus all -e WANDB_MODE=disabled \
  -v "$HOME/.cache/ghosttrial-ov/ov_data:/root/.local/share/ov" \
  -v "$PWD/logs_rl:/opt/gr00t/logs_rl" \
  -v "$PWD:/gt" -w /opt/gr00t ghosttrial-sonic:232 \
  "/isaac-sim/python.sh gear_sonic/eval_agent_trl.py \
     +checkpoint=${CKPT_IN_C} \
     +num_envs=1 +headless=True \
     ++export_onnx_only=True \
     ++manager_env.commands.motion.motion_lib_cfg.motion_file=${MOTION} \
     ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
     > /gt/logs/export_onnx.log 2>&1"
