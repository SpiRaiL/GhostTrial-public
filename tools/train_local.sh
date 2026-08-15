#!/usr/bin/env bash
# Fine-tune SONIC on our combo, on the local GPU.
#
#   tools/train_local.sh [motion_dir] [num_envs] [logfile]
#
# Defaults to take A3 — the captured combo with the spear throw flattened and the
# tucked arm swung clear of the thigh (see report §14).
#
# Two things that will silently waste a run if changed:
#   * the image must be ghosttrial-sonic:232. gear_sonic targets Isaac Lab 2.3.2;
#     3.0 removed SimulationCfg.physx and there is no patching around that.
#   * the whole command goes in ONE quoted string. The base image's ENTRYPOINT is
#     ["/bin/bash","-c"], so `docker run img /bin/bash -c '...'` runs nothing at all.
#
# Output is redirected to a file on the mounted volume rather than piped, because
# piping through tail buffers and loses everything if the run is killed.
set -euo pipefail
cd "$(dirname "$0")/.."

MOTION="${1:-/gt/data/motion_lib_capture/robot/a3}"
ENVS="${2:-1024}"
LOG="${3:-/gt/logs/train_a3.log}"
# resume from one of our own checkpoints instead of the stock SONIC release
CKPT="${CKPT:-sonic_release/last.pt}"

mkdir -p logs logs_rl "$HOME/.cache/ghosttrial-ov/ov_data"

# logs_rl MUST be mounted. The trainer writes its checkpoints to
# /opt/gr00t/logs_rl inside the container, and the container runs --rm, so
# without this the policy is deleted the moment the run stops.
exec docker run --rm --gpus all -e WANDB_MODE=disabled \
  -v "$HOME/.cache/ghosttrial-ov/ov_data:/root/.local/share/ov" \
  -v "$PWD/logs_rl:/opt/gr00t/logs_rl" \
  -v "$PWD:/gt" -w /opt/gr00t ghosttrial-sonic:232 \
  "/isaac-sim/python.sh gear_sonic/train_agent_trl.py \
     +exp=manager/universal_token/all_modes/sonic_release \
     +checkpoint=${CKPT} \
     num_envs=${ENVS} headless=True \
     ++manager_env.commands.motion.motion_lib_cfg.motion_file=${MOTION} \
     ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
     ++callbacks.model_save.save_frequency=${SAVE_EVERY:-200} \
     ++algo.config.save_interval=${SAVE_EVERY:-200} \
     > ${LOG} 2>&1"
