#!/usr/bin/env python3
"""Roll a trained policy out in Isaac and dump what the robot actually did.

Run inside the training container:

    /isaac-sim/python.sh /gt/tools/rollout_to_csv.py \
        --checkpoint <abs path to .pt> \
        --motion /gt/data/motion_lib_capture/robot/a3 \
        --out /gt/data/rollouts/a3_policy.csv --steps 600

Why this exists. The demo video needs the POLICY moving, not the reference, and
neither route through Isaac works here: enabling cameras segfaults Isaac Sim's
renderer on this machine, and eval_agent_trl.py's rollout is a `while True` that
only breaks on max_render_steps, so it hangs by default.

So: run the policy under Isaac for the physics, record joint angles and root pose
per step, and write them in the same BONES-SEED CSV format everything else here
speaks. tools/render_motion.py then renders it through MuJoCo and EGL, which is
proven and never touches Isaac's renderer.

Setup mirrors eval_agent_trl.py deliberately — same config merge, same model
construction, same checkpoint handling — because a policy fed subtly different
observations produces plausible-looking garbage with no way to tell.
"""

import argparse
import io
import os
import sys

sys.path.insert(0, "/opt/gr00t")
os.chdir("/opt/gr00t")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=700)
ap.add_argument("--num_envs", type=int, default=1,
                help="training uses 2048-4096; a 1-env rollout may not behave the same")
ap.add_argument("--config", default=None,
                help="config.yaml to use; defaults to the one beside the checkpoint. "
                     "Needed for the stock SONIC release, which ships weights only.")
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402
import omegaconf  # noqa: E402

# headless, and cameras OFF — enabling them is what segfaults
app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

import easydict  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gear_sonic import train_agent_trl  # noqa: E402
from gear_sonic.trl.trainer import ppo_trainer  # noqa: E402
from gear_sonic.trl.utils import common as trl_utils_common  # noqa: E402
from gear_sonic.utils import config_utils, obs_utils  # noqa: E402

config_utils.register_rl_resolvers()

# ── config: the checkpoint's own, exactly as eval_agent_trl merges it ─────────
# The saved config uses Hydra's own resolvers — ${hydra:runtime.choices.exp} and
# ${now:...}. Those only exist inside a @hydra.main run, and omegaconf resolves the
# WHOLE tree at once, so a single unresolvable key kills instantiation with
# "Unsupported interpolation type". Register stand-ins rather than chasing each key.
import datetime  # noqa: E402

omegaconf.OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.datetime.now().strftime(fmt), replace=True)
omegaconf.OmegaConf.register_new_resolver(
    "hydra", lambda _p: "manager/universal_token/all_modes/sonic_release", replace=True)

cfg_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
raw = open(cfg_path).read()
config = omegaconf.OmegaConf.load(io.StringIO(raw))
with omegaconf.open_dict(config):
    config.checkpoint = args.checkpoint
    config.num_envs = args.num_envs
    config.headless = True
    config.manager_env.commands.motion.motion_lib_cfg.motion_file = args.motion
    config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = "dummy"
    # Force the g1 encoder. encoder_sample_probs is {g1:1, teleop:1, smpl:1} — equal
    # weights — so every env is randomly assigned ONE of three encoders. At
    # num_envs=1 that is a 2-in-3 chance of evaluating in teleop mode, which tracks
    # head and hands from 3-point VR targets and drives the lower body from a
    # separate command the clip does not supply: the robot swings its arms with the
    # motion and never steps. That is what made stock SONIC "fail" too, and it is a
    # property of the run, not of the policy.
    for name in ("teleop", "smpl"):
        if name in config.manager_env.commands.motion.encoder_sample_probs:
            config.manager_env.commands.motion.encoder_sample_probs[name] = 0.0
    # Start every env at frame 0. Without this, forward_motion_samples() gives each
    # env a RANDOM start time inside the clip (commands.py:768 —
    # sample_time_steps(...) unless start_from_first_frame), so a single-env rollout
    # begins at a random phase of the motion and records the tail of it. At 2048 envs
    # only 3 travelled the full distance: the ones that happened to start near zero.
    with omegaconf.open_dict(config):
        config.manager_env.commands.motion.start_from_first_frame = True

device = "cuda:0"
args_cli = easydict.EasyDict({"headless": True, "enable_cameras": False,
                              "device": device, "kit_args": ""})

env = train_agent_trl.create_manager_env(config, device, args_cli)

module_dim_dict = getattr(config.algo.config, "module_dim", {})
env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]

# The universal-token policy has observation groups beyond policy/critic — the
# motion tokenizer among them — and their dims must be registered before the actor
# is built, or instantiation dies with "Missing key tokenizer".
example_obs = env.reset(flatten_dict_obs=False)
for key in env.env.observation_space:
    if key not in ("policy", "critic"):
        dims, names, total = obs_utils.get_group_term_obs_shape(example_obs, key)
        env.config["obs"]["group_obs_dims"][key] = dims
        env.config["obs"]["group_obs_names"][key] = names
        env.config["obs"]["obs_dims"][key] = total
        env.config["robot"]["algo_obs_dim_dict"][key] = total

meta_action_dim = env.config.get("meta_action_dim", None)
env.config["robot"]["actions_dim"] = (meta_action_dim if meta_action_dim
                                      else env.env.action_space.shape[-1])

policy = trl_utils_common.custom_instantiate(
    config.algo.config.actor, env_config=env.config, algo_config=config.algo.config,
    module_dim_dict=module_dim_dict, backbone_kwargs={}, _resolve=False).to(device)
value_model = trl_utils_common.custom_instantiate(
    config.algo.config.critic, env_config=env.config, algo_config=config.algo.config,
    module_dim_dict=module_dim_dict, backbone_kwargs={}, _resolve=False).to(device)
model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
sd = ckpt.get("policy_state_dict")
if "std" in model.policy.state_dict() and "log_std" in sd and "std" not in sd:
    sd["std"] = torch.exp(sd.pop("log_std"))
elif "log_std" in model.policy.state_dict() and "std" in sd and "log_std" not in sd:
    sd["log_std"] = torch.log(sd.pop("std"))
model.policy.load_state_dict(sd)
model.eval()
print(f"[rollout] loaded checkpoint from step {ckpt['state'].global_step}", flush=True)

# ── roll out, recording what the robot actually did ──────────────────────────
env.set_is_evaluating(True)
obs_dict = env.reset_all()
for k in obs_dict:
    obs_dict[k] = obs_dict[k].to(device)

robot = env.env.scene["robot"]
rows = []
# act_inference, NOT rollout(). policy.rollout() is the TRAINING data-collection
# path; act_inference is what the eval callback in im_eval_callback.py uses — the
# same eval that reports healthy tracking during training. Driving the robot with
# the training path made every policy, including stock SONIC, stand in place:
# the control walk travelled 0.09 m against its reference's 1.29 m.
#
# init_rollout() once before the loop, matching that callback. cur_dones and
# skip_episode_attnmask are part of the inference signature and are not optional —
# the policy builds an episode attention mask from them.
model.policy.init_rollout()
dones = torch.zeros(1, device=device)
with torch.no_grad():
    for step in range(args.steps):
        actions = model.policy.act_inference(
            obs_dict=obs_dict, cur_dones=dones, skip_episode_attnmask=True)
        actor_state = {"actions": actions, "obs": obs_dict, "obs_dict": obs_dict}

        if step == 0:
            start_xy = robot.data.root_pos_w[:, :2].clone()
        if step == args.steps - 1 or step % 100 == 0:
            trav = (robot.data.root_pos_w[:, :2] - start_xy).norm(dim=-1)
            print(f"[travel] step {step:4d}  envs={trav.numel()}  "
                  f"max={trav.max().item():.2f} m  mean={trav.mean().item():.2f} m  "
                  f"n>0.5m={(trav > 0.5).sum().item()}", flush=True)
        jp = robot.data.joint_pos[0, :29].cpu().numpy()
        rp = robot.data.root_pos_w[0].cpu().numpy()
        rq = robot.data.root_quat_w[0].cpu().numpy()          # w, x, y, z
        eul = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]]).as_euler("xyz", degrees=True)
        rows.append(np.concatenate([[step], rp * 100.0, eul, np.degrees(jp)]))

        if os.environ.get("GT_DEBUG_PHASE") and step == 0:
            tk = obs_dict.get("tokenizer")
            print(f"[obs] groups={list(obs_dict.keys())}", flush=True)
            mc = env.motion_command
            print(f"[start] motion_start_time_steps={mc.motion_start_time_steps[:4].tolist()} "
                  f"(should be all 0 if start_from_first_frame landed)", flush=True)
            if tk is not None:
                print(f"[obs] tokenizer shape={tuple(tk.shape)} first8={tk.flatten()[:8].tolist()}", flush=True)
        if os.environ.get("GT_DEBUG_PHASE") and step % 40 == 0:
            mc = env.motion_command
            print(f"[phase] step {step:4d} time_steps={int(mc.time_steps[0])} "
                  f"motion_id={int(mc.motion_ids[0])}", flush=True)
        res = env.step(actor_state)
        obs_dict, dones = res[0], res[2]
        dones = dones.float().reshape(-1)
        # Stop at the first termination. Running past it records the env
        # auto-resetting mid-clip, and running past the END OF THE MOTION records
        # the policy with nothing left to track — which is what made the first
        # videos show the robot walking off and falling over.
        if bool(dones.any()):
            print(f"[rollout] terminated at step {step}", flush=True)
            break
        if step % 100 == 0:
            print(f"[rollout] step {step}/{args.steps}", flush=True)

# Isaac reports world coordinates, and env 0's origin sits tens of metres from the
# world origin, so the raw trajectory renders far outside the camera. Shift XY so
# the rollout starts where the reference does; heading already matches.
rows = np.array(rows)
ref_xy = np.array([float(os.environ.get("GT_REF_X", 0.0)),
                   float(os.environ.get("GT_REF_Y", 0.0))])
rows[:, 1:3] -= rows[0, 1:3] - ref_xy

names = [n for n in robot.data.joint_names[:29]]
header = ("Frame,root_translateX,root_translateY,root_translateZ,"
          "root_rotateX,root_rotateY,root_rotateZ,"
          + ",".join(f"{n}_dof" for n in names))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savetxt(args.out, rows, delimiter=",", header=header, comments="", fmt="%.6f")
print(f"[rollout] wrote {args.out}  ({len(rows)} frames)", flush=True)
simulation_app.close()
