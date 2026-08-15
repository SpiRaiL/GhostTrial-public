#!/usr/bin/env python3
"""Cut the two reference windows into ONE continuous phrase for a performer.

    .venv/bin/python tools/make_reference_cut.py "<out dir>"

The two source windows are separate takes at different resolutions, one composited
inside arcade-cabinet artwork. To read as a single phrase they need:

  * the cabinet border cropped away (inner rect from isolate_character.py)
  * a common canvas, since the sources are 1280x720 and 640x480
  * the second window's long idle head trimmed - it holds a stance for 2.4 s before
    the descent starts, which would read as a dead stop mid-phrase
  * a short crossfade at the seam so the join reads as a transition, not a cut

Then the whole phrase again at half speed, because a mover learning a phrase from
video learns it from the slow pass.

Output: reference_phrase.mp4, ~29 s, well under any attachment limit.
"""

import os
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
TMP = "/tmp/refcut"
os.makedirs(TMP, exist_ok=True)

# Inner video rect of the arcade-cabinet composite, from isolate_character.py
X0, X1, Y0, Y1 = 172, 1110, 10, 712
W, H = 960, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
XFADE = 0.5

# Windows chosen from the measured segmentation in each raw/*/README.md.
# Part 2 starts 2.0 s into its window: frames 0-72 there are idle stance.
PARTS = [
    dict(src=f"{REPO}/raw/mk-bts-getoverhere/mk_bts_getoverhere.mp4",
         ss=120.0, dur=6.0, crop=f"crop={X1-X0}:{Y1-Y0}:{X0}:{Y0}",
         removelogo=None),
    dict(src=f"{REPO}/raw/mk-bts-uppercut/mk_bts_uppercut.mp4",
         ss=175.0, dur=4.0, crop=None,
         # The cyan "www.masterpesina..." watermark sits across the performer's
         # midsection, so delogo's rectangle would smear their body. removelogo
         # takes a pixel mask and only interpolates the text strokes themselves.
         # Mask built from the union of cyan pixels across the window, plus the
         # trailing white dots (which are not cyan), dilated 3px.
         removelogo=f"{REPO}/raw/mk-bts-uppercut/watermark_mask.png"),
]


def norm(part, path):
    """Crop, letterbox onto the common canvas, lock to 30 fps."""
    # removelogo runs first, on native resolution — the mask matches the source
    vf = ([f"removelogo=filename={part['removelogo']}"] if part.get("removelogo") else [])
    vf += ([part["crop"]] if part["crop"] else []) + [
        f"scale={W}:{H}:force_original_aspect_ratio=decrease",
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black",
        "fps=30,format=yuv420p",
    ]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(part["ss"]), "-t", str(part["dur"]), "-i", part["src"],
         "-an", "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "18", path],
        check=True)
    return path


a = norm(PARTS[0], f"{TMP}/a.mp4")
b = norm(PARTS[1], f"{TMP}/b.mp4")

# one continuous phrase, crossfaded at the seam
joined = f"{TMP}/joined.mp4"
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-i", b,
     "-filter_complex",
     f"[0][1]xfade=transition=fade:duration={XFADE}:offset={PARTS[0]['dur']-XFADE}",
     "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", joined],
    check=True)


def label(src, text, speed, path):
    vf = []
    if speed != 1.0:
        vf.append(f"setpts={1/speed}*PTS")
    vf.append(
        f"drawtext=fontfile={FONT}:text='{text}':x=28:y=24:fontsize=28:"
        f"fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=12")
    vf.append("fps=30,format=yuv420p")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an",
                    "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "20", path],
                   check=True)
    return path


segs = [label(joined, "ONE CONTINUOUS PHRASE", 1.0, f"{TMP}/s0.mp4"),
        label(joined, "SAME PHRASE  —  HALF SPEED", 0.5, f"{TMP}/s1.mp4")]

listfile = f"{TMP}/list.txt"
with open(listfile, "w") as fh:
    for s in segs:
        fh.write(f"file '{s}'\n")

out = f"{OUT}/reference_phrase.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", listfile, "-c", "copy", out], check=True)
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=width,height", "-of",
                "default=noprint_wrappers=1", out])
