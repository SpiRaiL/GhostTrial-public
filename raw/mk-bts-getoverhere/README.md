# MK1 behind-the-scenes — "Get Over Here" spear throw (video-lifting source)

Frame-by-frame reference of Daniel Pesina performing the Scorpion spear throw on
the Mortal Kombat digitising stage, cut out from the background.

**This whole folder is gitignored** (see `/.gitignore`) — it holds third-party
copyrighted video. Only `isolate_character.py` and this README are tracked.
Derive motion data from it; do not redistribute the frames.

## Source

| | |
|---|---|
| Video | "Mortal Kombat Behind-The-Scenes — Get Over Here Creation \| WTF Time" |
| YouTube | `fzWsWqPluus` — https://www.youtube.com/watch?v=fzWsWqPluus |
| Downloaded | 2026-08-07, 1280x720, 29.97 fps (30000/1001) |
| Window used | **2:00.0 – 2:06.0**, 180 frames, no resampling |

`video lifting` is a sanctioned data source for the hackathon. Note this is
documentary footage of a human actor on a stage — it is *not* ripped game
animation data, which the root README rules out.

## What's in the clip

The 6 s window is one continuous rep of the spear throw, roughly:

| frames | what |
|---|---|
| 001–057 | settled fighting stance, small idle adjustments |
| 061–089 | coil: torso rotates, throwing arm draws back and up |
| 093–137 | the throw — arm snaps forward, held at full extension |
| 141–180 | recovery back to stance |

**Relevance:** neither hero motion (MK uppercut, SF hadouken) is in this clip.
What it gives is the iconic MK forward-thrust silhouette from real reference —
structurally the closest thing in the repo to the hadouken's
wind-up → explosive forward thrust → hold-at-extension, and useful for styling
the exaggerated MK timing. Treat it as style reference, not as a retarget source.

## Files

```
mk_bts_getoverhere.mp4        full 2:19 source video
mk_bts_getoverhere_audio.mp3  the audio-only download this started from (see below)
frames_full/    frame_0001-0180.jpg  1280x720, the composited arcade-border frame
masks/          frame_0001-0180.png  8-bit matte, 938x702 (inner video rect)
character_alpha/frame_0001-0180.png  RGBA cutout, 938x702, absolute position kept
character_tight/frame_0001-0180.png  RGBA cutout cropped to the per-frame bbox
character_boxed/frame_0001-0180.jpg  same crop, background left intact
bboxes.csv      frame, source_time_s, x, y, w, h, area_px  (in 1280x720 coords)
contact_sheet.png  every 4th cutout, for eyeballing the whole take
preview_frames.mp4 / preview_cutout.mp4
```

Use `character_alpha/` for anything motion-related — it preserves absolute
position, so bbox translation across frames is real stage movement.
`character_tight/` re-centres per frame and destroys that.

## Reproducing

```bash
yt-dlp -f 'bv*[ext=mp4]+ba[ext=m4a]' -o mk_bts_getoverhere.mp4 \
    'https://www.youtube.com/watch?v=fzWsWqPluus'
ffmpeg -ss 120 -to 126 -i mk_bts_getoverhere.mp4 \
    -fps_mode passthrough -q:v 2 frames_full/frame_%04d.jpg
../../.venv/bin/python isolate_character.py     # ~3 min on CPU
```

Needs `rembg` + `onnxruntime` + `opencv-python-headless` + `pillow` in
`../../.venv` (already installed). First run downloads the isnet model
(179 MB) to `~/.u2net/`.

## Why segmentation and not background subtraction

The camera is locked off and the inner footage runs uncut from 27.9 s to
128.6 s, so a clean background plate looks easy. It isn't: **the actor never
steps off his mark**, so a per-pixel temporal median over any window — 6 s or
the full 100 s — bakes a ghost of him into the plate. There is no empty-stage
frame in the take. Hence per-frame matting (`rembg`, `isnet-general-use`,
which beat `u2net` on the hood and the extended arm).

Two matting quirks, both handled in `clean_mask()` and both worth knowing if
you re-tune it:

- His black trousers against the dark floor only score ~60–120 alpha, so a flat
  128 threshold **severs the legs** (it did, on frame 30). Fixed with hysteresis:
  grow from confident cores at 140 out through everything above 45, and drop
  weak blobs containing no core — that last rule is what rejects his cast
  shadow on the wall.
- Filling every enclosed hole **swallows the gap between the legs** on frames
  where the contact shadow bridges the two boots (it did, on frames 26 and 34).
  Only holes under 2000 px are filled.

## Known limitations

- A little contact shadow stays attached at the boots on some frames. It is
  connected to the body and low-impact; separating it cleanly is not worth it.
- The matte is per-frame with no temporal smoothing. It is stable in practice —
  after the fixes above, bbox centre moves at most 24 px/frame horizontally and
  14 px vertically, and no frame's area, width or height deviates >20/15/12%
  from its 5-frame local median — but there is mild edge shimmer.
- Source is VHS-era, heavily compressed and interlace-softened. Fine detail
  (fingers, the rope/spear prop) is mush in places.
- The camera slowly zooms in later in the take (after ~2:10). Irrelevant for
  this window, which is static, but it means `X0/X1/Y0/Y1` in the script are
  **not** valid for arbitrary other windows of the video.

## Aside: the `.mp3`

This started from `Mortal Kombat Behind-The-Scenes … WTF Time.mp3` in
`~/Downloads`, which was an audio-only file with no video stream — nothing to
extract frames from. It's kept here only as a record of where the request
started; `mk_bts_getoverhere.mp4` is the real asset.
