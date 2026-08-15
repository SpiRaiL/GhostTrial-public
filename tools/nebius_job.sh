#!/usr/bin/env bash
# Runs INSIDE the Nebius AI Job. Builds the SONIC environment from public sources,
# fine-tunes on our take, and ships checkpoints back to object storage.
#
# Why it assembles itself instead of running a prebuilt image: our training image is
# 40.5 GB and the uplink from the workstation is ~8.7 Mbps, so pushing it would take
# ten hours. Everything large here is public and pulls at datacentre speed —
# nvcr.io/nvidia/isaac-lab:2.3.2 needs no auth, and GR00T-WholeBodyControl is a
# public NVlabs repo. Only the SONIC checkpoint and our motion come from our bucket,
# because those are the two things that are not public.
#
# Env expected:
#   NB_KEY_ID / NB_KEY_SECRET   object-storage credentials
#   NB_BUCKET                   bucket name
#   RUN_NAME                    prefix for results, e.g. a11
#   MOTION_KEY                  object key of the motion_lib .pkl
#   NUM_ENVS, MAX_ITER, SAVE_EVERY
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y WANDB_MODE=disabled

# The Isaac Lab image has no bare `pip` on PATH and no git-lfs, and everything
# python must go through /isaac-sim/python.sh — that interpreter is the one with
# the environment gear_sonic needs. Bare `python3` here has no boto3.
PY_BIN=/isaac-sim/python.sh
echo "=== [1/5] tools ==="
apt-get update -qq && apt-get install -y -qq git-lfs
$PY_BIN -m pip install --quiet --no-input boto3

fetch() {  # key -> local path
  $PY_BIN - "$1" "$2" <<'PY'
import boto3, os, sys
boto3.client("s3", endpoint_url="https://storage.eu-north1.nebius.cloud:443",
             region_name="eu-north1",
             aws_access_key_id=os.environ["NB_KEY_ID"],
             aws_secret_access_key=os.environ["NB_KEY_SECRET"]
             ).download_file(os.environ["NB_BUCKET"], sys.argv[1], sys.argv[2])
print("  fetched", sys.argv[1])
PY
}

echo "=== [2/5] GR00T-WholeBodyControl from GitHub ==="
git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git /opt/gr00t
cd /opt/gr00t
# the robot assets are LFS objects; without them the MJCF/USD referenced by the env
# config resolve to pointer files and env construction fails
git lfs install --local
# only the robot assets, not the whole 5.7 GB of LFS objects in this repo
git lfs pull --include="gear_sonic/data/assets/**"
# Mirror docker/Dockerfile.isaaclab232 exactly. The package is the gear_sonic
# SUBDIRECTORY with a [training] extra, not the repo root — `pip install -e .` at
# /opt/gr00t fails. The directory must still be named gear_sonic, because its
# pyproject resolves the version via `attr: gear_sonic.version.VERSION`.
$PY_BIN -m pip install --no-cache-dir -e "gear_sonic/[training]"
# undeclared runtime imports on the training path; the pyproject omits them
$PY_BIN -m pip install --no-cache-dir \
  rich termcolor tyro tensordict open3d vector_quantize_pytorch

echo "=== [3/5] checkpoint + motion from object storage ==="
mkdir -p /opt/gr00t/sonic_release /opt/gr00t/motion
# CKPT_KEY lets a run resume from one of OUR checkpoints instead of the stock
# release, so a long run can continue where a previous one stopped.
fetch "${CKPT_KEY:-ckpt/sonic_release/last.pt}" /opt/gr00t/sonic_release/last.pt
fetch "${MOTION_KEY}" "/opt/gr00t/motion/$(basename "${MOTION_KEY}")"
ls -la /opt/gr00t/sonic_release /opt/gr00t/motion

echo "=== [4/5] train ==="
mkdir -p /opt/gr00t/logs_rl
# Checkpoints are pushed to the bucket as they appear rather than at the end: an AI
# Job's disk is gone when it exits, and a timeout would otherwise take the run with it.
( while sleep 300; do
    $PY_BIN - <<'PY' || true
import boto3, os, glob
s3 = boto3.client("s3", endpoint_url="https://storage.eu-north1.nebius.cloud:443",
                  region_name="eu-north1",
                  aws_access_key_id=os.environ["NB_KEY_ID"],
                  aws_secret_access_key=os.environ["NB_KEY_SECRET"])
run = os.environ["RUN_NAME"]
for f in glob.glob("/opt/gr00t/logs_rl/**/*.pt", recursive=True) + \
         glob.glob("/opt/gr00t/logs_rl/**/config.yaml", recursive=True):
    key = f"runs/{run}/" + os.path.relpath(f, "/opt/gr00t/logs_rl")
    try:
        if s3.head_object(Bucket=os.environ["NB_BUCKET"], Key=key)["ContentLength"] == os.path.getsize(f):
            continue
    except Exception:
        pass
    s3.upload_file(f, os.environ["NB_BUCKET"], key)
    print(f"  [sync] {key}", flush=True)
PY
  done ) &
SYNC=$!

$PY_BIN gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs="${NUM_ENVS:-4096}" headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/opt/gr00t/motion \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++callbacks.model_save.save_frequency="${SAVE_EVERY:-100}" \
  ++algo.config.save_interval="${SAVE_EVERY:-100}" \
  ++algo.config.num_learning_iterations="${MAX_ITER:-2000}" || TRAIN_RC=$?
  # ^ num_learning_iterations, NOT max_iterations. Hydra's ++ creates a key that
  # does not exist rather than failing, so the wrong name trains silently until the
  # job times out — which on a billed-by-wall-clock job means paying for the ceiling.

kill $SYNC 2>/dev/null || true
echo "=== [5/5] final sync ==="
$PY_BIN - <<'PY'
import boto3, os, glob
s3 = boto3.client("s3", endpoint_url="https://storage.eu-north1.nebius.cloud:443",
                  region_name="eu-north1",
                  aws_access_key_id=os.environ["NB_KEY_ID"],
                  aws_secret_access_key=os.environ["NB_KEY_SECRET"])
run = os.environ["RUN_NAME"]
n = 0
for f in glob.glob("/opt/gr00t/logs_rl/**/*", recursive=True):
    if os.path.isfile(f):
        s3.upload_file(f, os.environ["NB_BUCKET"],
                       f"runs/{run}/" + os.path.relpath(f, "/opt/gr00t/logs_rl"))
        n += 1
print(f"  uploaded {n} files to runs/{run}/")
PY
echo "JOB_DONE rc=${TRAIN_RC:-0}"
