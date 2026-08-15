# From the MK frames to G1 pose data — research + plan

Written 2026-08-07. Companion to `docs/magic_attack_pipeline.md` (the Mixamo path,
already working) and `raw/mk-bts-getoverhere/README.md` (the frames).

## TL;DR

There is an **NVIDIA-native video → G1 pipeline that lands in exactly the format we
already validated**, and it is built from the same retargeter that produced
BONES-SEED. It also fixes the two defects the Mixamo/GMR route left us with —
ground contact and joint-limit handling.

```
video ─► GEM-X ─► SOMA (77-joint) ─► soma-retargeter ─► G1 29-DoF CSV
                                                            │
                        convert_soma_csv_to_motion_lib.py ◄──┘   (we already use this)
                                     └─► motion_lib PKL ─► SONIC
```

## What the research turned up

### The main find — GEM-X + SOMA Retargeter (recommended)

| | |
|---|---|
| [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) | monocular video → **77-joint SOMA** motion (body + hands + face), world- and camera-space. Bundled 2D keypoint detector trained for SOMA joints. Apache 2.0 code; weights under the NVIDIA Open Model License. Checkpoint `gem_soma.ckpt` auto-downloads from HF. Python 3.10+/3.12, Torch 2.10+, CUDA 12.6+. |
| [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter) | SOMA BVH → **Unitree G1 29-DoF CSV**. GPU IK via **Newton + NVIDIA Warp**. Does proportional human→robot scaling, multi-objective IK with joint limits, **feet stabilisation to maintain ground contact**, per-DOF limit clamping. Python 3.12, Git LFS, NVIDIA GPU (Maxwell+), driver 545+. JSON config for retargeting/scaling/feet params. |

**Two facts that make this the right choice:**

1. **soma-retargeter is what made BONES-SEED.** Its README states plainly that the
   G1 motion data in SEED was retargeted with it. So anything we push through it is
   *distributionally identical* to SONIC's training data — the same solver, the
   same scaling, the same limit clamping. Fine-tuning on data produced a different
   way is a needless domain gap.
2. **It has explicit feet stabilisation.** That is precisely the defect measured in
   the Mixamo path: GMR's IK carries no ground-contact constraint and floated the
   robot 35.7–88.0 mm (see the progress report §4). We patched it with our own
   `tools/ground_motion.py`; NVIDIA solve it inside the retargeter.

GEM-X wires the two together directly — install `third_party/soma-retargeter`, then
`--retarget` on the demo emits `.bvh` and `.csv` alongside the render.

### The alternative we already have installed

GMR ships [`scripts/gvhmr_to_robot.py`](../vendor/GMR/scripts/gvhmr_to_robot.py) —
[GVHMR](https://github.com/zju3dv/GVHMR) extracts pose from monocular video, GMR
retargets to G1. Since GMR is already installed and working here, this is the
**fallback if GEM-X install fights us**. Downside: it inherits the same missing
ground constraint, so `ground_motion.py` stays in the loop, and the output is not
distribution-matched to BONES-SEED.

Also noted and rejected for now: [World-Coordinate Human Motion Retargeting via SAM
3D Body](https://arxiv.org/abs/2512.21573) — uses SAM 3D Body + the Momentum Human
Rig instead of a SLAM pipeline, explicitly positioned as avoiding WHAM/TRAM
complexity. Interesting, but it is not the format SONIC eats.

### How others have done it

| Work | Relevance |
|---|---|
| [VideoMimic](https://arxiv.org/html/2505.03729) — *Visual Imitation Enables Contextual Humanoid Control* | The reference real-to-sim-to-real result: casual monocular video → joint human+scene reconstruction → retarget to **Unitree G1** → RL in sim → real deployment. Stairs, slopes, sitting. First real-world context-aware humanoid policy learned from monocular human video. |
| [RPG: Robust Policy Gating for Smooth Multi-Skill Transitions in Humanoid Fighting](https://arxiv.org/abs/2604.21355) (Jun 2026) | **The closest prior art to GhostTrial.** Combat motions — punching, jumping, sword swing, kicking — captured from video, retargeted to G1, per-skill expert networks trained by imitation, then gated for transitions. Deployed on a real G1. Read this before designing the training run. |
| [HDMI](https://arxiv.org/html/2509.16757) — interactive whole-body control from human videos | Video → whole-body control with object interaction. |
| [HumanX](https://arxiv.org/pdf/2602.02473) | Agile, generalisable humanoid interaction skills from human videos. |
| [Tajima, *Making the Unitree G1 Dance from Video with GEAR-SONIC and GEM*](https://note.com/ryosuke_tajima/n/n1341bc889c4e) | A practitioner walkthrough of the exact stack we are proposing, on an **RTX 5060 Ti 16 GB** — so our 5080 16 GB is enough. Reported pain: TensorRT 10.13.3 build, a missing-GVHMR-module install gap, Torch/Blackwell version mismatch, SMPLX licensing. |

**The single most useful warning**, from that last one: GEM's demo *"is basically
designed for videos where only one person is visible in full body"*, and manually
cropping the footage to isolate one dancer was **"the hardest part of the whole
process."**

We have already done that work. `raw/mk-bts-getoverhere/character_alpha/` is a
single-person, full-body, background-removed cutout with absolute position
preserved. The hardest step of the reference workflow is behind us before we start.

## The plan

### Phase 0 · The move is "Get Over Here" — settled (2026-08-07)

Checking the actual rules changed this. The Martial Arts track asks for *"strikes,
kicks, throws, forms… the harder and more original the move, the higher it scores."*
**"Throws" is listed explicitly**, so the Scorpion spear throw is directly in scope —
and a coil-and-snap throw is both harder and more original than the hadouken's
two-handed palm thrust. The uppercut and hadouken were never requirements; they were
this project's own invented goals. See the README's competition block.

So the target is the spear throw. The only open question was the *source*, and the
answer is:

> **Perform it ourselves on camera.** The rules put the licensing burden on the
> entrant (*"You're responsible for your licenses"*), and the 1992 behind-the-scenes
> video belongs to Warner Bros/NetherRealm. The *move* is not copyrightable. Filming
> our own performance is still "lift motion from video" — a sanctioned source — with
> zero licensing exposure, and it documents cleanly in the required dataset
> methodology.

`raw/mk-bts-getoverhere/` therefore becomes **style and timing reference**: the
frame-accurate record of how Pesina actually coils, snaps and holds, used to direct
the performance and to check our own timing against. It is not training data. That
is a better use of it than pose estimation on VHS-grade footage was ever going to be.

### Phase 1 · Install GEM-X + soma-retargeter (local, RTX 5080, free)

```bash
cd ~/RC/competitions/GhostTrial/vendor
git clone --recursive https://github.com/NVlabs/GEM-X.git && cd GEM-X
pip install uv && uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e third_party/soma && (cd third_party/soma && git lfs pull)
bash scripts/install_env.sh
uv pip install -e third_party/soma-retargeter
```

Its own venv — **do not** merge into the project `.venv`. GEM-X wants Python 3.12 /
Torch 2.10 / CUDA 12.6; ours is Python 3.14 with Torch 2.13 for GMR. Keep them apart.

Gate: `python scripts/demo/demo_soma.py --video <their example> --retarget` produces
a `.csv`. Do not touch our footage until the bundled example works.

### Phase 2 · Build the input video GEM actually wants (local, free)

GEM's demo takes **`--video` only** — no image-folder input — so the 180 PNGs must
be re-encoded. Three things to get right:

1. **Feed the inner video rect, not the arcade border.** `frames_full/` is the
   composited 1280×720 frame with MK cabinet art down both sides. The real footage
   is the inner 938×702 region — that is what `masks/` and `character_alpha/` are
   already cropped to. Encoding the bordered frames would hand GEM two extra
   "people" made of dragon artwork.
2. **Decide cutout vs. original.** Try **`character_boxed/`** (background intact,
   cropped to the subject) *first*. GEM's detector was trained on natural images;
   a hard alpha matte on black is out of distribution and can hurt. Keep
   `character_alpha/` as the fallback if a second person or the mic stand confuses
   the detector.
3. **Upscale.** The subject is only 194–382 px wide, 364–457 px tall. Scale ~2× on
   encode to give the keypoint detector more pixels.

```bash
# 29.97 fps to match the source exactly — no resampling
ffmpeg -framerate 30000/1001 -i character_boxed/frame_%04d.jpg \
       -vf "scale=iw*2:ih*2:flags=lanczos" -c:v libx264 -crf 16 -pix_fmt yuv420p \
       /tmp/mk_spear_2x.mp4
```

### Phase 3 · Run GEM → SOMA → G1 (local, free)

```bash
python scripts/demo/demo_soma.py --video /tmp/mk_spear_2x.mp4 --retarget -s --verbose
```

- **`-s` / `--static_cam` is right here** — the digitising stage camera is locked
  off, so disabling visual odometry removes a whole failure mode. Drop `-s` for
  handheld phone footage in Phase 6.
- `--verbose` writes debug overlays; check the 2D keypoints land on the actor
  before trusting anything downstream.
- Output goes to `<output_root>/<video_name>/` — renders, tensors, and the
  retargeted `.bvh` + `.csv`.

### Phase 4 · Validate, with the checks that already exist (local, free)

This is where the existing tooling pays off. Point it at the new CSV:

| Check | Tool | Pass condition |
|---|---|---|
| Column schema vs a real BONES-SEED CSV | one-liner in the report §3 | identical |
| All 29 DOFs inside `jnt_range` | limit check | 0 violations, and note how many are *pinned* |
| Floor contact | `tools/ground_motion.py` (measure only) | float ≈ 0 mm — soma-retargeter should already deliver this. **If it does, that independently confirms the GMR float was a GMR problem.** |
| Eyeball it | `tools/render_motion.py` | looks like a spear throw |

Note the format fork: soma-retargeter may emit the **three-file** layout
(`joint_pos.csv`, `body_pos.csv`, `body_quat.csv`) rather than one flat CSV.
`convert_soma_csv_to_motion_lib.py` handles both — mode 1 for the directory,
mode 4 for a flat file. Check which you got before converting.

### Phase 5 · Into motion_lib (local, free)

```bash
.venv/bin/python \
  vendor/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input <gem_output_dir> --output data/motion_lib_mk_spear/robot --fps 30
```

Same command family as the Mixamo path. `--fps_source` only matters if GEM emits at
a rate other than the source 29.97.

### Phase 6 · Shoot the spear throw (local, free) — this is the deliverable

Match the reference footage deliberately, because the timing is the character of the
move. From `raw/mk-bts-getoverhere/README.md`, Pesina's rep is: settled stance →
coil (torso rotates, throwing arm draws back and up) → snap forward, hold at full
extension → recovery. Roughly 6 s for the whole rep at 29.97 fps.

Shooting notes, in priority order:

- **Static camera on a tripod**, locked off — matches the original stage setup and
  lets us keep `-s` (no visual odometry, one less failure mode).
- **Full body in frame at all times**, with headroom and floor visible. GEM's
  single-fully-visible-person constraint is the documented hard part; framing solves
  it at capture time instead of in post.
- **Plain, uncluttered background**, even lighting, no one else in shot.
- **Fitted clothing** — the reference performance is in baggy trousers and that is a
  known SMPL-fitting failure mode. Ours does not have to be.
- **60 fps or better** if the phone allows; downsample later, never upsample.
- **Several takes**, and vary them: mirrored (left-arm throw), a slower and a sharper
  rep, slightly different stance widths. Motion-tracking fine-tunes overfit hard to a
  single trajectory — this is the cheapest diversity available.
- **A short walk-in and walk-out** on at least one take. WBT-Bench includes a
  fundamentals check on walking and turning; having locomotion in our own data helps
  the policy keep it.

Read [RPG](https://arxiv.org/abs/2604.21355) before designing the capture — it is the
same problem (video-sourced punches and kicks on a real G1) and will inform how many
takes and how much variety to shoot.

## Scoring implications — what WBT-Bench actually rewards

Judging criterion #1 is *"SONIC's tracking reward, with penalties for flailing,
self-collision, and jitter, plus a fundamentals check: walking, turning."* Three
consequences worth designing around:

| Penalty | Our exposure | Action |
|---|---|---|
| **Self-collision** | Already measured: the Mixamo magic attack registers self-contact where the hands cup together. A spear throw is single-armed and far less prone to this — another point in its favour. | Run a self-collision count over the retargeted clip before training; MuJoCo reports it in `d.ncon` once floor contacts are filtered out. |
| **Flailing / jitter** | The pinned ankles in the Mixamo retarget are exactly the setup for instability. soma-retargeter's feet stabilisation and limit clamping should help. | Re-run the joint-limit and float checks on the new clip; treat any pinned ankle as a blocker. |
| **Fundamentals check (walking, turning)** | **The biggest under-appreciated risk.** Fine-tuning a general controller hard on one short clip is the classic recipe for catastrophic forgetting — we could nail the throw and lose the walking points. | Mix baseline BONES-SEED locomotion into the fine-tune set, keep the learning rate low, cap the step count, and **evaluate walking and turning before submitting**, not after. |

## Risks, specific to this footage

| Risk | Detail | Mitigation |
|---|---|---|
| **Source quality** | 1990s VHS-era, heavily compressed, soft. Fine detail on limbs is mush. | 2× upscale; accept that hands/feet will be noisy; `--verbose` overlays to judge. |
| **Costume** | Baggy trousers and a full mask. Loose clothing is a known SMPL-fitting failure mode — the model fits the *garment* silhouette, not the leg. | Expect leg-pose error; sanity-check knee angles against the video by eye. |
| **Props in frame** | A mic stand and stage clutter sit in the inner rect. | `character_boxed`/`character_alpha` already remove them. |
| **Subject size** | 364–457 px of ~700 px height. Workable, not generous. | Upscale; crop tighter if the detector struggles. |
| **Wrong move** | It is a spear throw, not an uppercut or hadouken. | Phase 0 — it is the test article. Do not let it become the deliverable. |
| **Install friction** | The practitioner report hit TensorRT, a GVHMR install gap, and a Torch/Blackwell mismatch. Our 5080 *is* Blackwell. | Separate venv; the CUDA 12.6 wheel index above; budget a session for this and fall back to GMR+GVHMR if it stalls. |
| **Licensing** | GEM weights are under the NVIDIA Open Model License, not Apache. SMPLX has its own terms. The MK footage remains third-party copyright. | Cite in the dataset docs; keep `raw/mk-bts-getoverhere/` gitignored (rules exist — but `git init` has not been run yet). |

## Why not just use the frames directly

Worth stating so nobody re-litigates it: the 180 PNGs are the *input* to pose
estimation, not pose data. Nothing downstream — not SONIC, not the retargeter —
consumes images. The mattes and bboxes matter because they solve GEM's
single-full-body-person constraint, which is the documented hard part. That is the
value the isolation work added, and it is already banked.

---

## Update, same day — the G1-direct route beat both alternatives

Before any capture arrives, a third option was tried and it wins on every metric
WBT-Bench penalises: **author the 29 DoF directly in the robot's own joint space**
(`tools/g1_author.py`), skipping the human rig and the retarget entirely.

| route | frames | pinned joints | self-collisions |
|---|---|---|---|
| Mixamo clip, retargeted (GMR) | 35 | 6 | 5 |
| Hand-authored on a human rig, retargeted | 81 | 11 | 805 |
| **Authored in G1 joint space** | **173** | **0** | **0** |

The middle row is the important negative: hand-animating a *human* and retargeting
made things **worse** than the Mixamo clip, because it inherits exactly the
human→G1 mismatch — a pose that is comfortable for a person buries the G1's bulky
shoulder and thigh links in its torso. Tuning the parameters barely moved it
(805 from 890). The mismatch is structural, not a tuning problem.

Authoring in joint space removes it by construction: values are clamped to the
MJCF's `jnt_range`, self-collision is measurable per keypose, and 29 DoF + root
*is* the BONES-SEED CSV format, so there is nothing to retarget.

Deliverables now in the repo:

```
data/g1_authored/spear_uppercut.csv     173 frames, 5.77 s, the full combo
data/g1_authored/spear_uppercut_M.csv   mirrored (also 0 / 0 / 0)
data/motion_lib_combo/robot/            both, converted for SONIC
```

**Conventions, measured not assumed** (this is the part that costs time if skipped):
+X forward / +Y left / +Z up; `hip_pitch` negative swings the knee forward;
`knee` positive flexes; and — the one that bit — **at zero `qpos` the arms point
straight forward and horizontal, not hanging down**, so `shoulder_pitch=0, elbow=0`
is already the throw extension and −90 puts the arm vertically overhead.

### This does not retire the capture

The authored combo is kinematically clean, not *natural*. Timing is keyed by hand
against the reference clips; a real martial artist's weight shifts, anticipation and
follow-through are what the tracking reward actually rewards. Treat the authored clip
as a **guaranteed-legal fallback and a pipeline proof**, and the commissioned capture
as the thing that wins on quality. If the capture lands well, lifted motion should
beat authored motion — and if it doesn't, there is already a submittable combo.
