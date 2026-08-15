# GhostTrial — Ghost Trial 03 (Martial Arts track)

[![GET OVER HERE — Scorpion's spear throw and uppercut on a Unitree G1](https://img.youtube.com/vi/1Oo8h655VMM/maxresdefault.jpg)](https://youtu.be/1Oo8h655VMM)

<p align="center"><b><a href="https://youtu.be/1Oo8h655VMM">▶ Watch the five-minute build (YouTube)</a></b></p>

| | |
|---|---|
| 🎬 Video | **[youtu.be/1Oo8h655VMM](https://youtu.be/1Oo8h655VMM)** — 5:51, chapters in the description |
| 🤖 Policy (ONNX) | **[huggingface.co/DaveRc/ghosttrial-g1-scorpion](https://huggingface.co/DaveRc/ghosttrial-g1-scorpion)** |
| 📊 Motion data | **[huggingface.co/datasets/DaveRc/ghosttrial-g1-scorpion-motion](https://huggingface.co/datasets/DaveRc/ghosttrial-g1-scorpion-motion)** |
| 📄 Start here | **[`docs/submission.md`](docs/submission.md)** — deliverables, before/after, what is not in this repo |

Goal: teach a Unitree G1 a **combat move the stock SONIC model can't do**, by
fine-tuning the SONIC controller on custom motion data.

**Chosen move: the Scorpion spear throw → uppercut, as one continuous combo.**
Commissioned from a martial artist on camera, lifted to pose data, retargeted to the
G1. Both reference clips come from the same Daniel Pesina digitising session:
[spear throw](https://www.youtube.com/watch?v=fzWsWqPluus) at 2:00–2:06 and
[uppercut](https://www.youtube.com/watch?v=bskmWBAqBZI) at 2:53–2:59.
See [`docs/video_to_pose_plan.md`](docs/video_to_pose_plan.md) and
[`docs/upwork_capture_brief.md`](docs/upwork_capture_brief.md).

The combo — rather than either move alone — is deliberate: the track scores
"harder and more original", the transition is the part a single-pose clip can't
teach, and a longer trajectory is a better hedge against the fine-tune overfitting
to one second of motion.

## Competition parameters (verified against the site 2026-08-07)

Competition: https://www.ultimatebots.com/hackathon — **Ghost Trial 03**

| | |
|---|---|
| Deadline | **Aug 16, 2026, 11:59pm PT — hard.** Winners Aug 18. |
| Tracks | Martial Arts · Performance Arts · Experimental. **$1,000 each**, $3,000 total. |
| Entry | Free, open worldwide, solo or teams up to 4. |
| Hardware | **No robot needed — everything runs in simulation.** League embodiments: Unitree G1, Booster K1, Agibot X2, LeRobot SO-101. |
| Deliverables | ONNX-exported fine-tuned policy · dataset docs + methodology · reproducible training config · sim demo video · before/after baseline comparison |
| Data sources | Kimodo text-to-motion · BONES-SEED · LAFAN · **lift motion from video** · capture your own |

**Martial Arts track, verbatim:** *"Strikes, kicks, throws, forms — teach the G1
combat movement the stock model can't perform. The harder and more original the move,
the higher it scores."*

**Judging:**
1. **WBT-Bench score** — *"SONIC's tracking reward, with penalties for flailing,
   self-collision, and jitter, plus a fundamentals check: walking, turning."*
2. Difficulty & originality of the chosen move
3. Execution quality and reliability
4. Pipeline cleanliness — dataset, config, reproducibility

**Licensing, verbatim:** *"You keep ownership of everything you create — dataset,
code, model."* and *"You're responsible for your licenses. Build only on models and
data whose terms permit it; some datasets are free only for non-commercial or
small-company use."*

### What this means for us — three corrections to the original plan

1. **Nobody asked for an uppercut or a hadouken.** Those were this project's own
   invented goals, not requirements. The track wants *strikes, kicks, throws, forms*.
   **"Throws" is listed explicitly** — the spear throw is squarely in scope, and it
   scores better on "harder and more original" than a two-handed palm thrust does.
2. **The footage is the licensing problem, not the move.** Moves aren't
   copyrightable; the 1992 behind-the-scenes video is Warner Bros'/NetherRealm's, and
   the rules put the licensing burden on us. Performing the throw ourselves on camera
   removes the issue entirely and still yields "lift motion from video", a sanctioned
   source. `raw/mk-bts-getoverhere/` stays as **style reference**, not training data.
3. **WBT-Bench scores things we have already measured badly on.** Self-collision is
   penalised — and our Mixamo clip already registers self-contact where the hands cup
   together. Flailing and jitter are penalised — that is what the pinned ankles risk.
   And the **fundamentals check (walking, turning)** means the fine-tuned policy must
   not forget how to walk: fine-tuning hard on one 1.2 s clip is the fastest way to
   lose those points. Mix baseline motions into the fine-tune, keep the learning rate
   low, and evaluate walking before submitting.

## Folder layout

```
raw/                          human-skeleton source data (needs retargeting to G1)
  cmu-boxing/                 CMU mocap boxing trials, AMC + ASF (license: free, any use)
                              13_17 13_18 14_01-03 15_04 15_05 15_13 17_10
                              boxing sets include uppercuts; 15_* also has dance/misc
  bandai-namco/               Bandai Namco Research mocap, BVH, 3000+ clips
                              (CC BY-NC-ND 4.0 — OK to train on non-commercially,
                              do NOT redistribute retargeted copies)
                              fighting clips: dataset-1 punch/kick/slash
  mixamo/                     Mixamo FBX, skeleton only, mixamorig 65-joint rig:
                              standing_2h_magic_attack_04 (hadouken source),
                              uppercut, great_sword_crouching (crouch donor)
                              — see raw/mixamo/README.md for the splice plan
retargeted/                   already in Unitree G1 format
  g1-kungfu/                  ASAP/PHC pickles (root_trans_offset, pose_aa, dof[23],
  g1-lafan-fight/             root_rot, smpl_joints, fps=30, contact_mask) — from
                              huggingface.co/datasets/openhe/g1-retargeted-motions (MIT)
                              kungfu: Hooks_punch, Horse-stance_punch, Roundhouse_kick,
                              Side_kick, Bruce_Lee_pose, Horse-stance_pose
                              lafan: LAFAN1 fight1/fightAndSports1 (Ubisoft LAFAN1
                              license: non-commercial research)
  bones-seed/                 BONES-SEED, downloaded + extracted (72 GB on disk).
                              g1/csv/{date}/*.csv — 142,220 motions, 29-DoF @ 120 fps,
                              a DIFFERENT format from the pickles above.
                              See bones-seed/NOTES.md — including why it contains
                              no uppercut and no hadouken.
tools/                        seed_search.py — search BONES-SEED annotations
docs/                         notes, competition materials
.venv/                        joblib+numpy for the pickles, pandas+pyarrow+
                              huggingface_hub for BONES-SEED
```

## The two hero motions — where the data comes from

### Uppercut (MK style: deep crouch → explosive rising punch)
1. **CMU boxing** (downloaded, `raw/cmu-boxing/`) — real boxing mocap incl. uppercuts.
   Free for any use. Needs AMC→BVH/SMPL conversion + retargeting to G1.
2. ~~**BONES-SEED**~~ — **dead end for the uppercut.** Downloaded and searched
   (`retargeted/bones-seed/`): "uppercut" / "upward punch" / "rising punch" /
   "jab" return **0 hits across all 142,220 motions**, and the entire Martial
   Arts category is 20 clips of comedy face-punching and stance transitions.
   The only usable material is `shadow_boxing_R` (11 takes) — unsegmented free
   boxing, i.e. the same manual-trimming job as the CMU trials.
3. **Mixamo** (adobe login required, browser) — `raw/mixamo/uppercut.fbx` is
   downloaded. Its wind-up is a shallow boxing stance, not the MK deep crouch,
   so it gets spliced with `great_sword_crouching.fbx`; see
   `raw/mixamo/README.md`. Free to use in projects.
4. **Kimodo** synthetic variants for the exaggerated MK look (see below).

### Hadouken (no real mocap exists → synthesize)
1. **Kimodo** (NVIDIA, Apache 2.0, github.com/nv-tlabs/kimodo) — text-to-motion that
   outputs **Unitree G1 motion directly** (also NPZ / MuJoCo CSV / AMASS). ~17 GB VRAM,
   2–5 s per generation → run on a Nebius GPU instance. Hackathon-endorsed.
   Prompt sketch: "fighting stance; pulls both hands back to the right hip, cups them
   together gathering energy, then explosively thrusts both palms forward at chest
   height, holding the extended pose" — generate N seeds, keep the best.
2. **Mixamo** — `raw/mixamo/standing_2h_magic_attack_04.fbx` is downloaded and is
   **the current primary hadouken source** (single clip, real mocap base, no GPU
   needed) — this is the move being built first.
3. **BONES-SEED** — no forward palm thrust exists in it either ("fireball",
   "blast", "both hands forward" → 0 hits). What it does have is
   `idle_right_to_spell_idle_two_hands_R`, 72 takes of neutral → two-handed
   spell idle across 72 actors: a usable wind-up/recovery and style-variation
   library bracketing the Mixamo thrust. See `retargeted/bones-seed/NOTES.md`.

Not available: the SLMP kickboxing dataset (arXiv 2603.01294) has explicit uppercut
mocap but no public release — contacting the authors is a long shot before Aug 16.

## Pipeline (recommended)

1. Get SONIC starter scripts from the hackathon repo; baseline = SONIC checkpoint
   (finetune, don't train from scratch — see arXiv 2511.07820).
2. Retarget human-skeleton sources to G1 with GMR (general motion retargeting,
   github.com/YanjieZe/GMR — supports BVH/SMPL → G1) or Mink
   (github.com/kevinzakka/mink, what openhe used). CMU AMC→BVH first
   (Blender or amc2bvh).
3. Convert to **motion_lib PKL** — the one format SONIC trains on
   (`root_trans_offset`, `pose_aa`, `dof`, `root_rot`, `fps`), produced by
   `vendor/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`.
   That script reads BONES-SEED `g1/csv/` natively and handles the 120→30 fps
   downsample itself (`--fps 30 --fps_source 120`), so no manual reconciliation
   is needed. Then filter with `filter_and_copy_bones_data.py` (drops ~8.7%
   the G1 can't perform).
4. Segment/trim: isolate single uppercut reps from the boxing trials (they're long
   free-boxing takes); mirror left/right for augmentation.
5. Fine-tune on Nebius (watch out: use the correct tenant profile — the
   rc-spike-soft-surface jobs failed cross-tenant), export ONNX, record
   before/after sim video.

## Data access

BONES-SEED is gated: accept its licence with your own Hugging Face account, then
install a fine-grained token with `canReadGatedRepos` at
`~/.cache/huggingface/token` (mode 600) — `huggingface_hub` finds it
automatically, no login step needed. Nebius credentials are unrelated to this and
are not used for dataset access.

## Sources
- Hackathon: https://www.ultimatebots.com/hackathon
- BONES-SEED: https://huggingface.co/datasets/bones-studio/seed (SONIC paper: https://arxiv.org/abs/2511.07820)
- Kimodo: https://github.com/nv-tlabs/kimodo · https://research.nvidia.com/labs/sil/projects/kimodo
- G1 retargeted motions (MIT): https://huggingface.co/datasets/openhe/g1-retargeted-motions
- CMU mocap: https://mocap.cs.cmu.edu (boxing: subjects 13/14/15/17)
- Bandai Namco mocap: https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
- LAFAN1: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- KungfuBot (martial-arts G1 control reference): https://arxiv.org/abs/2506.12851
- SLMP kickboxing dataset (unreleased): https://arxiv.org/abs/2603.01294
- Mixamo: https://www.mixamo.com (search: Uppercut, Standing 2H Magic Attack)
