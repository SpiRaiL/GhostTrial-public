---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - motion-capture
  - humanoid
  - unitree-g1
  - retargeting
  - motion-tracking
pretty_name: GhostTrial — spear throw into an uppercut, retargeted to a Unitree G1
size_categories:
  - n<1K
---

# GhostTrial motion data — a spear throw into an uppercut, on a Unitree G1

The motion behind
[DaveRc/ghosttrial-g1-scorpion](https://huggingface.co/DaveRc/ghosttrial-g1-scorpion):
a two-part combat phrase — spear throw, pull back, crouch into a rising uppercut —
captured from a human performer and carried all the way to the G1's 29 degrees of
freedom.

Pipeline, tools and full write-up:
**https://github.com/SpiRaiL/GhostTrial-public**

The capture, the retarget and the trained result, on video: **https://youtu.be/1Oo8h655VMM**

## Where it came from

A movement phrase written as a brief and filmed by a commissioned performer
(`docs/upwork_capture_brief.md` in the GitHub repo), across several camera angles
and two shoots. Monocular video → 3D human pose → human skeleton in BVH →
retargeted to the G1 → corrected for self-collision, foot contact and joint limits
→ converted to the motion_lib format SONIC trains on.

**The performer's video is not published**, only the motion derived from it. The
1992 Mortal Kombat footage that inspired the move is style reference only; it never
entered training and is not distributed here.

## What is in it

| folder | what it is |
|---|---|
| `data/motion_lib_capture/robot/` | the pickles SONIC trains on, one folder per take. **`t12/T12_beats.pkl` is the one that trained the published policy** |
| `data/gemx_g1_retimed/` | the G1 takes as CSV, 29 DoF plus root — retimed, the working format |
| `data/gemx_g1_raw/` | the same takes before retiming |
| `data/human_bvh/`, `data/human_bvh_edited/` | the performer's own skeleton, as solved and as hand-corrected |
| `data/gemx_output/` | per-take pose solve and the retarget straight out of it |
| `data/csv_frozen/` | frozen reference poses: standing, idle, a walking control, and the crouch seeds |
| `data/capture_analysis/`, `data/capture02_analysis/` | 2D keypoints and per-take rankings from reviewing both shoots |
| `data/g1_authored/` | one phrase authored by hand in G1 joint space, the legal fallback and the first thing that trained |

## How the takes are named

Order of work, not quality:

- **A2–A11** — retarget revisions. A9 is the reference the diagnosis work is built
  on; A10 is the take whose feet left the floor; A11 is faster and lower.
- **S1, B1, CTRL, CTRL2** — standing, idle and walking controls, used to check the
  policy has not forgotten how to stand or walk.
- **T1–T12** — target rebuilds after the crouch turned out to be the problem. **T12**
  is the final one: the phrase with a 0.5 s lean and a 0.5 s crouch hold feathered
  in.

## Known limits

- One performer, one phrase. This is a hackathon dataset, not a corpus.
- The target still asks for airborne frames the G1 cannot hold — the trained policy
  plants instead (97% double support against the target's 58%).
- Solved from monocular video, so depth is inferred rather than measured. The
  angle-45 takes are the most reliable.

## Licence

CC BY 4.0. Derived from footage commissioned for this project, published with the
performer's work credited as the source of the motion.
