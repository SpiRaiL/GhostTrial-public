#!/usr/bin/env bash
# Submit a SONIC fine-tune to Nebius AI Jobs.
#
#   NB_KEY_ID=... NB_KEY_SECRET=... tools/nebius_submit.sh a11 \
#       motions/a11/A11_faster_deeper.pkl [PLATFORM] [PRESET] [HOURS]
#
# tools/nebius_job.sh is the thing that actually runs, injected straight into the
# container with --inject-file LOCAL:CONTAINER. The first attempt passed it as
# base64 in an env var and decoded it from an --args shell one-liner; the CLI does
# not keep that one-liner intact, so the container died with
# "/usr/bin/echo: cannot execute binary file". --args now carries a single path
# with no spaces, which is the form the jobs that work in this tenant use.
#
# DRIVER_CAPS: training runs fine on the default compute,utility — Isaac logs
# VkResult ERROR_INCOMPATIBLE_DRIVER and carries on headless. RENDERING does not:
# it dies with "Vulkan 1.1 is not supported", because no graphics capability was
# granted. Eval-with-video therefore needs DRIVER_CAPS=all. Same trap as the local
# `docker run --gpus all` runs, which also grant compute,utility only.
#
# --timeout is the cost ceiling, not a guess at the runtime: an AI Job bills for
# wall-clock, so the timeout is what bounds the spend if training runs long.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?run name, e.g. a11}"
MOTION_KEY="${2:?motion object key}"
# JOB_SCRIPT selects train (default) or eval; CKPT_KEY/CFG_KEY are eval-only
JOB_SCRIPT="${JOB_SCRIPT:-tools/nebius_job.sh}"
PLATFORM="${3:-gpu-h100-sxm}"
PRESET="${4:-1gpu-16vcpu-200gb}"
HOURS="${5:-3}"
PROJECT="${NB_PROJECT:-project-e00rvhqzpr0059t6qgkgbn}"
BUCKET="${NB_BUCKET:-rc-ghosttrial}"
: "${NB_KEY_ID:?}" "${NB_KEY_SECRET:?}"

export PATH="$HOME/.nebius/bin:$PATH"
nebius ai job create --parent-id "$PROJECT" \
  --image nvcr.io/nvidia/isaac-lab:2.3.2 \
  --platform "$PLATFORM" --preset "$PRESET" \
  --timeout "${HOURS}h" \
  --env NVIDIA_DRIVER_CAPABILITIES="${DRIVER_CAPS:-compute,utility}" \
  --env NB_KEY_ID="$NB_KEY_ID" \
  --env NB_KEY_SECRET="$NB_KEY_SECRET" \
  --env NB_BUCKET="$BUCKET" \
  --env RUN_NAME="$RUN" \
  --env MOTION_KEY="$MOTION_KEY" \
  --env NUM_ENVS="${NUM_ENVS:-4096}" \
  --env MAX_ITER="${MAX_ITER:-2000}" \
  --env SAVE_EVERY="${SAVE_EVERY:-100}" \
  --env CKPT_KEY="${CKPT_KEY:-}" \
  --env CFG_KEY="${CFG_KEY:-}" \
  --env EPISODES="${EPISODES:-4}" \
  --env RENDER_STEPS="${RENDER_STEPS:-600}" \
  --inject-file "$JOB_SCRIPT":/opt/job.sh \
  --container-command /bin/bash \
  --args /opt/job.sh
