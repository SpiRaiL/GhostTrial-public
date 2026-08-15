#!/usr/bin/env bash
# Roll a trained policy out under physics and dump the result as a BONES-SEED CSV.
#
#   tools/rollout_local.sh <checkpoint.pt> <motion_dir> <out.csv> [steps] [logfile]
#
# Physics only — no cameras, so this runs on a 16 GB card where the renderer will
# not. The CSV then goes through tools/check_rollout_contact.py and
# tools/render_motion.py (MuJoCo/EGL), which is how the videos get made.
#
# Same container rules as train_local.sh: image ghosttrial-sonic:232, and the whole
# command in ONE quoted string because the base ENTRYPOINT is ["/bin/bash","-c"].
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?checkpoint path}"
MOTION="${2:?motion dir, container path e.g. /gt/data/motion_lib_capture/robot/b1}"
OUT="${3:?output csv, container path e.g. /gt/data/rollouts/b1_policy.csv}"
STEPS="${4:-200}"
LOG="${5:-/gt/logs/rollout.log}"
CKPT_IN_C="/opt/gr00t/${CKPT#./}"

# The stock SONIC release ships a config.yaml whose _target_ paths still say
# `groot.*`, the package's old name — importing it raises ModuleNotFoundError.
# Point CONFIG at one of our own run configs (same architecture, current names)
# to roll the stock weights out.
CFG_ARG=""
[ -n "${CONFIG:-}" ] && CFG_ARG="--config ${CONFIG}"

exec docker run --rm --gpus all -e WANDB_MODE=disabled \
  -e GT_DEBUG_PHASE="${GT_DEBUG_PHASE:-}" \
  -v "$HOME/.cache/ghosttrial-ov/ov_data:/root/.local/share/ov" \
  -v "$PWD/logs_rl:/opt/gr00t/logs_rl" \
  -v "$PWD:/gt" -w /opt/gr00t ghosttrial-sonic:232 \
  "/isaac-sim/python.sh /gt/tools/rollout_to_csv.py \
     --checkpoint ${CKPT_IN_C} \
     --motion ${MOTION} \
     --out ${OUT} \
     --steps ${STEPS} ${CFG_ARG} \
     > ${LOG} 2>&1"
