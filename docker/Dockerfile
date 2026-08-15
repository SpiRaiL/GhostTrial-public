# SONIC training environment for GhostTrial.
#
# Base is the Isaac Lab image already built for the Nebius soft-surface challenge —
# it saves a ~47 GB Isaac Sim install and is known to work on this box's RTX 5080.
#
#   Isaac Sim  6.0.0
#   Isaac Lab  3.0.0     <-- NOTE: GR00T-WholeBodyControl targets 2.3.2. If the
#                            training entrypoint breaks on an API change, that is
#                            the first thing to suspect, and the fix is to pin a
#                            2.3.2 Isaac Lab base rather than to patch gear_sonic.
#   Python     3.12 (via /isaac-sim/python.sh)
#
# gear_sonic's pyproject reads its version with `attr: gear_sonic.version.VERSION`,
# so the source directory MUST still be called `gear_sonic` at install time —
# copying it under any other name fails with ModuleNotFoundError.
#
# Build:
#   docker build -f docker/Dockerfile -t ghosttrial-sonic:latest .
# Smoke test:
#   docker run --rm --gpus all -v $PWD:/gt ghosttrial-sonic:latest \
#     /isaac-sim/python.sh /opt/gr00t/gear_sonic/train_agent_trl.py --help

FROM cr.eu-north1.nebius.cloud/e00dcvz58twsb7qwbz/isaac-lab-test:latest

ENV PYTHONUNBUFFERED=1 \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=Y \
    GR00T_PATH=/opt/gr00t

COPY vendor/GR00T-WholeBodyControl /opt/gr00t

WORKDIR /opt/gr00t
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e "gear_sonic/[training]" 2>&1 | tail -40

# gear_sonic's pyproject does not declare every runtime import it makes.
# Found by running the trainer and reading the ModuleNotFoundErrors one at a time.
# A scan of gear_sonic's imports against the image showed ~25 undeclared modules.
# Most belong to teleop/camera/deploy subsystems the trainer never touches
# (depthai, pyrealsense2, unitree_sdk2py, xrobotoolkit_sdk, pygame, lerobot, ...).
# These are the ones on the training import chain:
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
      rich termcolor tyro tensordict open3d

# --- Isaac Lab 2.3 -> 3.0 compatibility ---------------------------------------
# gear_sonic targets Isaac Lab 2.3.2, this base is 3.0.0. First break:
# `isaaclab.utils.noise.AdditiveUniformNoiseCfg` was renamed to `UniformNoiseCfg`.
# Verified an exact drop-in: UniformNoiseCfg defaults to operation='add', which is
# precisely what the Additive variant meant. 5 YAML config files reference it.
# If further 2.3->3.0 breaks appear, stop patching and pin an Isaac Lab 2.3.2 base.
RUN grep -rl "AdditiveUniformNoiseCfg" /opt/gr00t/gear_sonic \
    | xargs -r sed -i "s/isaaclab.utils.noise.AdditiveUniformNoiseCfg/isaaclab.utils.noise.UniformNoiseCfg/g" \
 && echo "patched AdditiveUniformNoiseCfg -> UniformNoiseCfg"

WORKDIR /gt
