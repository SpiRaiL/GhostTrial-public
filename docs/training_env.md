# SONIC training environment — local smoke test

2026-08-07/08. Purpose: get `train_agent_trl.py` to start cleanly **before** paying
for a Nebius GPU, because the real threat to a $25 budget is setup, not training.

## Result in one line

**SONIC fine-tuning runs end to end on the local RTX 5080, on our authored combo,
for $0 of Nebius.** Verified: `Loaded 2 motions with a total length of 11.440s and
574 frames`, motion keys `['spear_uppercut', 'spear_uppercut_M']`, PPO iterating at
~155-185 steps/s with mean reward rising 8.07 → 9.55 → 10.31 over iterations 8-10.

Use **`docker/Dockerfile.isaaclab232`** (Isaac Lab 2.3.2). The other Dockerfile,
built on the Nebius-challenge Isaac Lab 3.0 base, is kept only as the record of why
3.0 does not work.

## What is verified working

| Step | Status |
|---|---|
| Docker image builds on the existing Nebius-challenge Isaac Lab base | ✅ |
| GPU passthrough to the RTX 5080 (driver 595.84), torch 2.10 + CUDA | ✅ |
| `import isaaclab`, `import gear_sonic` | ✅ |
| SONIC checkpoint downloaded (`sonic_release/last.pt`, 469 MB) | ✅ |
| Hydra resolves `+exp=manager/universal_token/all_modes/sonic_release` | ✅ |
| `+checkpoint=`, `num_envs=`, `headless=` overrides accepted | ✅ |
| **Our authored combo accepted as `motion_lib_cfg.motion_file`** | ✅ |
| **`smpl_motion_file=dummy` accepted** — no 30 GB SMPL download needed | ✅ |
| Isaac Lab AppLauncher starts headless, writes a run config | ✅ |
| Environment construction | ✅ (on 2.3.2) |
| Policy construction, checkpoint load | ✅ |
| **PPO training loop, rewards increasing** | ✅ |

## The Isaac Lab 3.0 dead end (why Dockerfile.isaaclab232 exists)

The first base image tried was **Isaac Lab 3.0.0 / Isaac Sim 6.0.0**; `gear_sonic` targets **2.3.2**.
Two breaks found, in order:

1. `isaaclab.utils.noise.AdditiveUniformNoiseCfg` → renamed `UniformNoiseCfg`.
   **Patched** — verified an exact drop-in, since `UniformNoiseCfg` defaults to
   `operation='add'`, which is what the Additive variant meant. 5 YAML files.
2. `SimulationCfg.physx` **no longer exists**. Isaac Lab 3.0 moved off PhysX, so
   this is not a rename — the whole physics-config surface changed.

Break 2 is where patching stops being sensible: re-plumbing physics config for a
different engine is a rabbit hole with no end, and any divergence silently changes
the simulation the policy trains in.

## What fixed it

**`nvcr.io/nvidia/isaac-lab:2.3.2` pulls straight from NGC, no auth needed.** That
is exactly the version in the repo's badge, and every 3.0 incompatibility vanished.
`docker/Dockerfile.isaaclab232` builds on it.

### The second blocker: Git-LFS meshes

On 2.3.2 the run then died with:

    Failed to open layer @/tmp/IsaacLab/usd_*/configuration/pelvis.tmp.usd@

Cause: the repo was cloned `--depth 1` **without git-lfs**, so all 105 G1 `*.STL`
meshes were ~130-byte LFS pointers and Isaac Lab's URDF→USD conversion had nothing
to convert. git-lfs is not installed on this box, so `tools/fetch_lfs.py` resolves
the pointers over plain HTTP via `media.githubusercontent.com` (works for any
public repo). 1275 of 1279 objects fetched; the 4 failures are aarch64/x86 `.so`
symlinks in the deploy-only Unitree SDK and are irrelevant to training.

    .venv/bin/python tools/fetch_lfs.py vendor/GR00T-WholeBodyControl NVlabs/GR00T-WholeBodyControl main

**Run this after any fresh clone of GR00T-WholeBodyControl.**

## Undeclared dependencies (version-independent, keep these)

`gear_sonic`'s pyproject does not declare everything it imports. A scan of its
imports against the image found ~25 missing modules; most belong to teleop, camera
and deploy subsystems the trainer never touches (`depthai`, `pyrealsense2`,
`unitree_sdk2py`, `xrobotoolkit_sdk`, `pygame`, `lerobot`, `pytorch3d`, ...).

These are on the **training** import chain and must be installed:

```
rich  termcolor  tyro  tensordict  open3d  vector_quantize_pytorch
```

Found by running the trainer and reading `ModuleNotFoundError`s one at a time —
`rich`, then `open3d`, then `vector_quantize_pytorch` (needed by the universal-token
policy's quantizer, so it only surfaces after the environment builds).

## Other gotchas

- **`WANDB_MODE=disabled`** is required or the run dies at `wandb.init` with
  "No API key configured", after Isaac has already started.
- The base image's `ENTRYPOINT` is `["/bin/bash","-c"]`, so pass the command as a
  **single string**; `docker run img /bin/bash -c '...'` silently runs nothing.
- `gear_sonic`'s pyproject reads its version via `attr: gear_sonic.version.VERSION`,
  so the source directory must still be named `gear_sonic` when pip installs it.
- Installing `gear_sonic[training]` downgrades numpy to 1.26.4 while Isaac Lab wants
  ≥2. Both still import, so it appears benign — but note it if odd numeric behaviour
  shows up.

## The smoke test command

```bash
# once, after any fresh clone:
.venv/bin/python tools/fetch_lfs.py vendor/GR00T-WholeBodyControl NVlabs/GR00T-WholeBodyControl main

docker build -f docker/Dockerfile.isaaclab232 -t ghosttrial-sonic:232 .

docker run --rm --gpus all -e WANDB_MODE=disabled \
  -v ~/.cache/ghosttrial-ov/ov_data:/root/.local/share/ov \
  -v $PWD:/gt -w /opt/gr00t ghosttrial-sonic:232 \
  '/isaac-sim/python.sh gear_sonic/train_agent_trl.py \
     +exp=manager/universal_token/all_modes/sonic_release \
     +checkpoint=sonic_release/last.pt \
     num_envs=16 headless=True \
     ++manager_env.commands.motion.motion_lib_cfg.motion_file=/gt/data/motion_lib_combo/robot/g1_authored \
     ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy'
```

**Do not pipe the container's output through `tail`** — it buffers, and if you kill
the run you lose everything. Redirect to a file on the mounted volume
(`> /gt/logs/smoke.log 2>&1`) and watch that instead.

Success looks like:

```
Loading motion data from /gt/data/motion_lib_combo/robot/g1_authored...
Current motion keys: ['spear_uppercut', 'spear_uppercut_M']
Loaded 2 motions with a total length of 11.440s and 574 frames.
...
 Learning iteration 10
 Computation: 185 steps/s   Mean rewards: 10.30827   Iteration time: 2.07s
```

A sample iteration with the full tracking-error breakdown is saved at
`logs/iteration_sample.txt`.

Anything less, read the traceback bottom-up: a `ModuleNotFoundError` is a missing
dependency (add it), an `AttributeError` on an `isaaclab.*` symbol is a version
mismatch (change the base image), and a USD/`Failed to open layer` error means the
LFS meshes are still pointers.
