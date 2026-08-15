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
# Keep a copy of everything and ship it with the results: one eval job completed in
# 4 minutes having produced no uploads, and the log API returned nothing for it, so
# there was no way to tell what it did. A self-captured log removes that dependency.
exec > >(tee -a /tmp/job_output.log) 2>&1
set -x
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

echo "=== [3/5] trained checkpoint + its config + motion ==="
mkdir -p /opt/gr00t/evalckpt /opt/gr00t/motion
# eval_agent_trl.py reads its config from BESIDE the checkpoint (checkpoint.parent/
# config.yaml), which is also why no +exp= is passed. Both must land together.
fetch "${CKPT_KEY}" /opt/gr00t/evalckpt/model.pt
fetch "${CFG_KEY}"  /opt/gr00t/evalckpt/config.yaml
fetch "${MOTION_KEY}" "/opt/gr00t/motion/$(basename "${MOTION_KEY}")"
CKPT_PATH=/opt/gr00t/evalckpt/model.pt
ls -la /opt/gr00t/evalckpt /opt/gr00t/motion

echo "=== [4/5] rollout under physics (no renderer needed) ==="
# Runs OUR rollout, not upstream eval: Nebius AI Jobs expose no graphics (Vulkan
# fails with "Graphics plugins not available" even at DRIVER_CAPS=all), so Isaac
# cannot render here either. Physics-only state comes back as a CSV and gets drawn
# with MuJoCo on the workstation, which is the pipeline that already works.
#
# It runs here rather than locally only because the 16 GB card is busy training;
# a single-env Isaac instance needs ~3 GB it does not have.
mkdir -p /opt/gr00t/tools
fetch "code/rollout_to_csv.py" /opt/gr00t/tools/rollout_to_csv.py
fetch "code/g1_columns.py"     /opt/gr00t/tools/g1_columns.py
# timeout, because Isaac frequently hangs in simulation_app.close() after the work
# is done ("USD stage detach not called"). The CSV is written before shutdown, so a
# hang would otherwise strand a finished result until the job timeout killed the
# whole container — sync included.
timeout 900 $PY_BIN /opt/gr00t/tools/rollout_to_csv.py \
  --checkpoint "${CKPT_PATH}" \
  --motion /opt/gr00t/motion \
  --out /opt/gr00t/rollout.csv \
  --steps "${RENDER_STEPS:-400}" \
  --num_envs "${NUM_ENVS:-1}" \
  --config /opt/gr00t/evalckpt/config.yaml || EVAL_RC=$?

echo "=== [5/5] sync ==="
$PY_BIN - <<'PYEOF'
import boto3, os
s3 = boto3.client("s3", endpoint_url="https://storage.eu-north1.nebius.cloud:443",
                  region_name="eu-north1",
                  aws_access_key_id=os.environ["NB_KEY_ID"],
                  aws_secret_access_key=os.environ["NB_KEY_SECRET"])
run = os.environ["RUN_NAME"]; n = 0
for f in ("/tmp/job_output.log", "/opt/gr00t/rollout.csv"):
    if os.path.isfile(f):
        s3.upload_file(f, os.environ["NB_BUCKET"], f"eval/{run}/" + os.path.basename(f)); n += 1
print(f"  uploaded {n} files to eval/{run}/")
PYEOF
echo "JOB_DONE rc=${EVAL_RC:-0}"
