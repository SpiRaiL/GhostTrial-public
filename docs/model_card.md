---
license: other
license_name: nvidia-open-model-license
license_link: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/LICENSE
base_model: nvidia/GR00T-WholeBodyControl
tags:
  - robotics
  - humanoid
  - unitree-g1
  - motion-tracking
  - onnx
  - isaac-lab
---

# GhostTrial — Scorpion spear throw into an uppercut, on a Unitree G1

A fine-tune of NVIDIA's SONIC whole-body controller that performs a two-part
combat phrase the stock controller cannot: a spear throw, a pull back, and a
crouch into a rising uppercut, as one continuous motion under full physics.

Entered in **Ghost Trial 03** (Ultimate Bots hackathon, Martial Arts track).
Code, data and the full pipeline: **https://github.com/SpiRaiL/GhostTrial-public**

## Files

| file | what it is |
|---|---|
| `model_step_003750_g1.onnx` | the encoder trained on G1-space motion — the one this pipeline drives |
| `model_step_003750_encoder.onnx` | motion encoder |
| `model_step_003750_decoder.onnx` | policy decoder |
| `config.yaml` | the training run exactly as it ran |
| `model_config.yaml` | model dimensions, written by the exporter |

## How it was trained

Fine-tuned from the SONIC release checkpoint — not trained from scratch — on one
motion: our own capture, retargeted to the G1 and rebuilt twelve times.

| | |
|---|---|
| Base | SONIC release checkpoint (GR00T-WholeBodyControl) |
| Environments | 4096 parallel, Isaac Lab 2.3.2, one H100 on Nebius |
| Checkpoint | step 3750 |
| Actor / critic LR | 2e-5 / 1e-3, gamma 0.99, seed 0 |
| Motion | `data/motion_lib_capture/robot/t12` in the GitHub repo |
| Export | `tools/export_onnx.sh <checkpoint> /gt/data/motion_lib_capture/robot/t12` |

## Data

A movement phrase written as a brief and filmed by a commissioned performer,
lifted to 3D pose from monocular video, retargeted to the G1's 29 DoF, then
corrected across eleven revisions for self-collision, foot contact and joint
limits before any GPU time was spent. The motion data is in the GitHub repo; the
performer's video is not published.

The 1992 Mortal Kombat behind-the-scenes footage that inspired the move is style
reference only. It never entered training and is not distributed.

## Before and after

Same reference motion, stock SONIC against this fine-tune, both under full
physics in Isaac Lab:

| | stock | this model |
|---|---|---|
| Pelvis height range over the phrase | 74.3–78.7 cm (4 cm) | 46.3–79.3 cm (33 cm) |
| The crouch the move is built on | never happens | performed |
| Drift from start over 14 s | 0 cm | 4 cm |

The stock controller stays upright and marks time. This one drives into the
stance, drops into the crouch and carries the arm through.

## Limitations

- It holds **97% double support** where the target asks for 58%: it plants where
  the reference leaves the floor, so the airborne part of the uppercut is damped.
- Trained on one phrase with standing, idle and walking control takes alongside
  it. Walking and turning were not separately evaluated against this checkpoint.
- Simulation only. It has never run on a physical G1.

## Licence

Derived from SONIC model weights, which are under the **NVIDIA Open Model
License** — that licence governs this derivative too. The pipeline code in the
GitHub repo is ours.
