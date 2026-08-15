# MK behind-the-scenes — uppercut (style reference)

Daniel Pesina performing the uppercut on the Mortal Kombat digitising stage. Same
shoot as [`../mk-bts-getoverhere/`](../mk-bts-getoverhere/), so the stance and body
language already match between the two clips — which is what makes the
spear-throw → uppercut combo believable as one sequence.

**This folder is third-party copyrighted video. Style and timing reference only —
not training data.** The commissioned capture
([`docs/upwork_capture_brief.md`](../../docs/upwork_capture_brief.md)) is the actual
source; see the licensing note in the root README.

## Source

| | |
|---|---|
| Video | "Scorpion (Daniel Pesina) Motion Capture for Mortal Kombat — Behind the Scenes" (Kombatology) |
| YouTube | `bskmWBAqBZI` — https://www.youtube.com/watch?v=bskmWBAqBZI |
| Downloaded | 2026-08-07, 640x480, 30 fps, 9:59 total |
| Window used | **2:53.0 – 2:59.0**, 180 frames, no resampling |

Note the title says "motion capture" — it is not. The 1992 game used
**digitisation** (filming actors, converting to 2D sprites); MK4 was the first in
the series to use real motion capture. There is no 3D data of this performance
anywhere, which is why it has to be re-performed.

## Segmentation

Measured by thresholding the costume's saturation per frame and tracking the top of
the figure (the stage is unsaturated grey, the costume is strongly orange):

| frames | t (from 2:53) | beat |
|---|---|---|
| 0–72 | 0.0–2.4 s | settled guard stance |
| 72–96 | 2.4–3.2 s | **descent** — sinking into the crouch |
| 96–115 | 3.2–3.8 s | **deep crouch**, held near the floor |
| 115–144 | 3.8–4.8 s | **drive up** into the rising punch |
| 144–155 | 4.8–5.2 s | extension held, arm overhead |
| 155–180 | 5.2–6.0 s | settle back to stance |

The head drops from row ~180 to row ~285 of 480 at the bottom of the crouch — the
sink is the dominant feature of the move, and it is much deeper than a boxing
uppercut.

## Files

```
mk_bts_uppercut.mp4    full 9:59 source video
frames_full/           frame_0001-0180.jpg, 640x480
```

Regenerate the frames with:

```bash
ffmpeg -y -ss 173.0 -i mk_bts_uppercut.mp4 -t 6.0 -vsync 0 -q:v 2 frames_full/frame_%04d.jpg
```

Unlike the spear-throw folder there is no character isolation here yet — it was not
needed, because the plan moved to authoring in G1 joint space
(`tools/g1_author.py`) and commissioning a real performance, rather than lifting
pose from this footage.
