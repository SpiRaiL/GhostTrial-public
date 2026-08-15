# Ghost Trial 03 — submission summary

**Video:** https://youtu.be/1Oo8h655VMM

**Track:** Martial Arts · **Embodiment:** Unitree G1 · **Move:** the Scorpion spear
throw into an uppercut, as one continuous phrase.

Everything here is built from a movement phrase we commissioned and filmed
ourselves, lifted to pose data from monocular video, retargeted to the G1, and used
to fine-tune the SONIC controller. The 1992 Mortal Kombat footage is style
reference only — it is not in this repo and it never entered training.

## Where each deliverable is

| asked for | here |
|---|---|
| ONNX-exported fine-tuned policy | `policy/model_step_003750_{g1,encoder,decoder}.onnx`, also published as a model on Hugging Face &mdash; `policy/README.md` is its card |
| Dataset docs + methodology | `docs/video_to_pose_plan.md`, `docs/upwork_capture_brief.md`, `docs/magic_attack_pipeline.md`, and this file |
| Reproducible training config | `policy/config.yaml` (the run as trained), `docs/training_env.md`, `docker/Dockerfile.isaaclab232`, `tools/nebius_job.sh`, `tools/train_local.sh` |
| Sim demo video | **https://youtu.be/1Oo8h655VMM** (5:51, chapters in the description) |
| Before/after baseline comparison | below, and in the video |

## The policy

Fine-tuned from the SONIC release checkpoint on our own motion, not trained from
scratch.

| | |
|---|---|
| Checkpoint | step 3750, 4096 parallel environments, on Nebius (H100) |
| Actor LR | 2e-5 · critic 1e-3 · gamma 0.99 · seed 0 |
| Training motion | `data/motion_lib_capture/robot/t12/T12_beats.pkl` |
| Export | `tools/export_onnx.sh <checkpoint> /gt/data/motion_lib_capture/robot/t12` |

`_g1` is the encoder trained on G1-space motion, which is what this pipeline
produces; `_encoder`/`_decoder` are the matching halves of the actor. The SMPL and
teleop encoders from the same export are not included — nothing here uses them.

## Before and after

The same reference motion, driven by the stock SONIC release and by the fine-tune,
both under full physics in Isaac Lab.

| | stock | fine-tuned |
|---|---|---|
| Pelvis height range over the phrase | 74.3–78.7 cm (4 cm) | 46.3–79.3 cm (33 cm) |
| The crouch the move is built on | never happens | performed |
| Drift from start position over 14 s | 0 cm | 4 cm |

The stock controller stays upright and marks time; the fine-tune drives into the
stance, drops into the crouch and carries the arm through. Neither falls, so the
difference is the move itself rather than stability.

Against its own target, the final policy holds 97% double support where the target
asks for 58% — it chooses stability over the target's airborne frames, which is the
honest limit of this entry and the next thing to fix.

## How the motion was made

1. **Capture.** A movement phrase written as a brief (`docs/upwork_capture_brief.md`)
   and filmed by a commissioned performer, several angles, two shoots.
2. **Video to pose.** Monocular 3D human pose per take, then a human skeleton in
   BVH: `data/human_bvh/`, hand-corrected in `data/human_bvh_edited/`.
3. **Retarget to the G1.** `data/gemx_g1_raw/` → retimed in `data/gemx_g1_retimed/`,
   with self-collision clearing, foot re-seating and per-frame joint-limit checks
   (`tools/clear_self_collision.py`, `tools/flatten_feet.py`,
   `tools/check_g1_motion.py`).
4. **Feasibility.** Static and dynamic checks against the support polygon before
   spending GPU time: `tools/contact_feasibility.py`, `tools/dynamic_feasibility.py`.
5. **motion_lib.** `data/motion_lib_capture/robot/*` — the one format SONIC trains
   on. CSV out of `tools/bvh_to_bones_csv.py`, then SONIC's own
   `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`.
6. **Fine-tune, evaluate, export.** `tools/nebius_job.sh`, `tools/nebius_eval_job.sh`,
   `tools/export_onnx.sh`.

Takes are named in order: A2–A11 are retarget revisions, S1/B1/CTRL are the
standing, idle and walking controls, and T1–T12 are the target rebuilds. T12 is the
one that trained.

## Licensing and what is deliberately absent

- **Our capture** — commissioned for this project; the motion data derived from it
  is published here. The performer's video is not.
- **Mortal Kombat footage** — Warner Bros./NetherRealm. Style reference only, never
  training data, not in this repo. `raw/mk-bts-*/README.md` says which clips and
  where they came from.
- **Mixamo** — free to use, not to redistribute. `raw/mixamo/README.md` names the
  clips; fetch them from mixamo.com with an Adobe account.
- **BONES-SEED, LAFAN1, Bandai Namco, CMU** — searched and documented in the
  README; the first three are non-commercial or no-redistribution, so no copies or
  retargets of them are published here.
- **SONIC / GR00T-WholeBodyControl** — NVIDIA, public repo, used as the pretrained
  base. Not vendored here; `docker/` builds the environment that fetches it.

`docs/PUBLISHED.md` lists exactly what was copied out of the working repo and what
was held back.
