#!/usr/bin/env python3
"""Chapter list for the YouTube cut, read out of the blend that was rendered.

    tools/chapter_links.py [video_id] [--markdown]

Times come from the level-title strips in the EDIT scene, so they match the file
that was uploaded rather than a plan of it. Without a video id it prints the plain
description block (YouTube turns those timestamps into chapters by itself); with
one it prints deep links, for the README, the submission form or a post.

    tools/chapter_links.py dQw4w9WgXcQ --markdown
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLEND = HERE.parent / "presentation" / "ghosttrial_presentation.blend"
BLENDER = "/snap/bin/blender"
CH_TITLE = 5

DUMP = '''
import bpy, json
e = bpy.data.scenes["EDIT"]
rows = [[int(s.frame_final_start), s.channel, s.text]
        for s in e.sequence_editor.strips if s.type == "TEXT"]
print("JSONSTART" + json.dumps({"rows": rows, "fps": e.render.fps}))
'''

# what the plate says -> what a viewer scrubbing the bar wants to read
RENAME = {
    "GET OVER HERE": "Get Over Here",
    "MORTAL KOMBAT": "Mortal Kombat, 1992",
    "FIRST BLOOD": "First Blood — finding the footage",
    "PRACTICE MODE": "Practice Mode — one move, end to end",
    "OG FOOTAGE AS MOCAP": "OG footage as mocap",
    "THE DANCER": "The Dancer — commissioning the capture",
    "SECOND TAKE": "Second Take — the reshoot",
    "GHOST IN THE MACHINE": "Ghost in the Machine — video to skeleton",
    "RETARGET": "Retarget — onto the G1",
    "TRAINING, TRAINING, TRAINING": "Training, training, training",
    "TEST YOUR MIGHT": "Test Your Might — making the soundtrack",
    "PHYSICS BOSS MODE": "Physics Boss Mode",
    "FINAL BOSS": "Final Boss",
}
# beats whose title strip was cut but which are still worth a chapter
EXTRA = {298: "The Move", 308: "The Pit"}
SKIP = {"TO BE CONTINUED", "A move becomes a legend"}


def chapters():
    r = subprocess.run([BLENDER, "--background", str(BLEND), "--python-expr", DUMP],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("JSONSTART"):
            data = json.loads(line[len("JSONSTART"):])
            break
    else:
        print(r.stdout[-600:], file=sys.stderr)
        raise SystemExit(f"could not read {BLEND}")

    fps = data["fps"]
    out = {}
    for frame, channel, text in sorted(data["rows"]):
        if channel != CH_TITLE or text in SKIP:
            continue
        sec = (frame - 1) // fps
        name = RENAME.get(text.strip())
        if name and sec not in out:
            out[sec] = name
    out.update({s: n for s, n in EXTRA.items() if s not in out})
    return sorted(out.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", nargs="?", help="the YouTube id, once it is up")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    for sec, name in chapters():
        stamp = f"{sec // 60}:{sec % 60:02d}"
        if not args.video_id:
            print(f"{stamp} {name}")
        elif args.markdown:
            print(f"- [{stamp} {name}](https://youtu.be/{args.video_id}?t={sec})")
        else:
            print(f"{stamp} {name} — https://youtu.be/{args.video_id}?t={sec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
