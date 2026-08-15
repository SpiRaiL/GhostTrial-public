#!/usr/bin/env bash
# Render a trained policy performing the move — submission slot 4 (demo video).
#
#   tools/render_policy.sh <checkpoint.pt> <motion_dir> [num_envs]
#
# Same Hydra rules as tools/export_onnx.sh: no +exp=, and keys that base_eval does
# not declare need a leading +.
#
# num_envs drives VRAM hard once cameras are on — upstream's own note is 64 envs
# ~= 23 GB, so on a 16 GB card keep this small. Videos land in
# <checkpoint dir>/renderings/ckpt_<n>/.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-$(ls -t logs_rl/TRL_G1_Track/manager/universal_token/all_modes/*/last.pt | head -1)}"
MOTION="${2:-/gt/data/motion_lib_capture/robot/a3}"
ENVS="${3:-4}"
CKPT_IN_C="/opt/gr00t/${CKPT#./}"

echo "rendering from: $CKPT  (motion $MOTION, ${ENVS} envs)"
exec docker run --rm --gpus all -e WANDB_MODE=disabled \
  -v "$HOME/.cache/ghosttrial-ov/ov_data:/root/.local/share/ov" \
  -v "$PWD/logs_rl:/opt/gr00t/logs_rl" \
  -v "$PWD:/gt" -w /opt/gr00t ghosttrial-sonic:232 \
  "/isaac-sim/python.sh gear_sonic/eval_agent_trl.py \
     +checkpoint=${CKPT_IN_C} \
     +num_envs=${ENVS} +headless=True \
     ++render_results=True \
     ++algo.config.eval.save_videos=True \
     ++algo.config.eval.num_eval_episodes=8 \
     ++algo.config.eval.save_goal_reached_only=False \
     ++algo.config.eval.video_save_prob=1.0 \
     ++manager_env.commands.motion.motion_lib_cfg.motion_file=${MOTION} \
     ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
     > /gt/logs/render_policy.log 2>&1"
