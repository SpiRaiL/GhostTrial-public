# Mixamo source animations

Downloaded from https://www.mixamo.com (Adobe login, browser-only — no API).
All three are **FBX, skeleton only, no skin**, standard `mixamorig:*` joint names
(65 joints). They need retargeting to G1 the same way the CMU/Bandai data does
(GMR or Mink — see the top-level README pipeline).

Mixamo assets are free to use in projects; keep them in `raw/` and do not
redistribute the retargeted copies as standalone Mixamo derivatives.

| File | Mixamo asset name | Role |
|---|---|---|
| `standing_2h_magic_attack_04.fbx` | Standing 2H Magic Attack 04 | **Hadouken** — the two-handed forward palm thrust. Primary source; this is the move we build first. |
| `uppercut.fbx` | Uppercut | **Uppercut** — rising punch, but from a fairly upright boxing stance. |
| `great_sword_crouching.fbx` | Great Sword Crouching | Deep crouch pose. Not a move on its own — the intended donor for the MK-style *deep* wind-up that `uppercut.fbx` lacks. |

## Converted

`bvh/standing_2h_magic_attack_04.bvh` — via `tools/fbx_to_bvh.py` (headless
Blender, `/snap/bin/blender`; snap isn't on the default PATH). Metres,
`mixamorig:` prefix stripped, 65 joints, 30 fps, 101 frames, in place — the
root never translates, so there's no locomotion to reconcile.

## Magic attack — segmentation

From `tools/clip_phases.py` (world-space hand midpoint relative to the hips):

| frames | t | phase |
|---|---|---|
| 1–22 | 0.00–0.70 s | idle, arms drifting up |
| **23** | 0.73 s | **charge** — hands most tucked (fwd −0.03, i.e. at the hip line), separation closing |
| **29** | 0.93 s | **peak hand speed, 5.84 m/s** — the thrust |
| **41** | 1.33 s | **full extension** — fwd 0.754 m, hands cupped (separation 0.13 m) |
| 41–80 | 1.33–2.67 s | static hold, hand speed ~0.1 m/s |
| 81–101 | 2.70–3.37 s | recovery to idle |

The move itself is **frames 23–41, 0.60 s**. Everything after 45 is a ~1.5 s
dead hold that should be trimmed to ~0.3 s before training, or it teaches the
policy to freeze.

⚠️ The hips drop from 0.97 m to 0.78 m between frames 21 and 33 — a 19 cm CoM
drop driven into the thrust. That is the part most likely to destabilise the G1
and the thing to watch first in the retarget.

## The uppercut = crouch + uppercut plan

The MK uppercut is a deep crouch that explodes into a rising punch. Mixamo's
`Uppercut` starts from a shallow boxing stance, so on its own it reads as a
regular punch on the G1. The plan is to splice:

1. Retarget both clips to G1 independently.
2. Take the descent + hold from `great_sword_crouching` as the wind-up.
3. Take the drive + extension from `uppercut` as the strike.
4. Blend across the transition (the frame where the hips reverse direction is
   the natural seam), then time-warp so the rising phase is faster than the
   descent — that acceleration asymmetry is what makes it read as MK.
5. Check knee/hip limits and CoM against the G1 model after splicing; the
   great-sword crouch is deeper than the G1's joint range in places.

Deferred — the hadouken (`standing_2h_magic_attack_04`) goes through the
pipeline first as the simpler single-clip case.
