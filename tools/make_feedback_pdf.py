#!/usr/bin/env python3
"""Build the reshoot feedback PDF for the performer.

    .venv/bin/python tools/make_feedback_pdf.py <out.pdf>

Renders an A4 document from the hand-picked pairs in data/sync_pairs.json — cover
page with the headline notes and the filming changes, then three beats per page,
original beside actor with the note underneath.

Light theme on purpose: this is an attachment someone may print, not a screen report.
Laid out in HTML and printed via headless Chrome, which handles pagination properly.
"""

import base64
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/data/upwork_package_02/1_pose_feedback.pdf"
FRAMES = "/tmp/seq"

pairs = json.load(open(f"{REPO}/data/sync_pairs.json"))


def b64(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


beats = []
for i, p in enumerate(pairs, 1):
    r, a = f"{FRAMES}/r{i}.png", f"{FRAMES}/a{i}.png"
    if not (os.path.exists(r) and os.path.exists(a)):
        raise SystemExit(f"missing frame images for beat {i} — run tools/make_sequence.py first")
    beats.append((i, b64(r), b64(a), (p.get("comment") or "").strip()))

CARDS = "\n".join(
    f'''<div class="beat">
      <div class="num">{i}</div>
      <div class="ims"><figure><img src="{ri}"><figcaption>ORIGINAL</figcaption></figure>
                       <figure><img src="{ai}"><figcaption>YOU</figcaption></figure></div>
      <div class="note">{c}</div>
    </div>''' for i, ri, ai, c in beats)

HTML = f"""<!doctype html><html><head><meta charset=utf-8><style>
@page {{ size: A4; margin: 14mm 12mm; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font:11pt/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
        color:#14181d; background:#fff; }}
h1 {{ font-size:21pt; margin:0 0 4pt; }}
h2 {{ font-size:13pt; margin:18pt 0 6pt; padding-bottom:3pt; border-bottom:1px solid #d7dde3; }}
.sub {{ color:#5d6873; font-size:10pt; margin:0 0 14pt; }}
.big {{ border:1px solid #d7dde3; border-left:4px solid #c0392b; border-radius:4px;
        padding:9pt 12pt; margin:8pt 0; background:#fbf7f7; }}
.big b {{ color:#a5301f; }}
.film {{ border:1px solid #d7dde3; border-left:4px solid #1f6f8b; border-radius:4px;
         padding:9pt 12pt; margin:8pt 0; background:#f5fafc; }}
.slow {{ border:1px solid #d7dde3; border-left:4px solid #1e8449; border-radius:4px;
         padding:9pt 12pt; margin:8pt 0; background:#f4faf6; }}
.slow b {{ color:#166a3a; }}
.slow p {{ font-size:10pt; }}
ul {{ margin:6pt 0 0 14pt; padding:0; }} li {{ margin:3pt 0; }}
.beat {{ display:grid; grid-template-columns:22pt 1fr; gap:7pt; align-items:start;
         border:1px solid #e2e7ec; border-radius:5px; padding:7pt; margin-bottom:7pt;
         break-inside:avoid; page-break-inside:avoid; }}
.num {{ font-size:16pt; font-weight:800; color:#1f6f8b; line-height:1; }}
.ims {{ display:flex; gap:8pt; }}
.ims figure {{ margin:0; }}
.ims img {{ height:168pt; width:auto; border:1px solid #ccd4db;
              border-radius:3px; display:block; }}
.ims figcaption {{ font-size:7.5pt; color:#6b7681; letter-spacing:.06em; margin-top:2pt; }}
.note {{ grid-column:2; font-size:10pt; margin-top:4pt; }}
.pagebreak {{ page-break-before: always; }}
.foot {{ margin-top:16pt; padding-top:6pt; border-top:1px solid #d7dde3; color:#6b7681; font-size:9pt; }}
</style></head><body>

<h1>Reshoot notes — spear throw into uppercut</h1>
<p class="sub">14 moments through the phrase, the original beside your take, with a note on each.
The original is 1990s behind-the-scenes footage, so it is soft — the shapes are what matter, not the detail.<br><b>Short version: shoot it much slower, get the positions exact, and ignore the timing — I can fix timing afterwards, positions I cannot.</b></p>

<div class="big">
  <b>The three that matter most</b>
  <ul>
    <li><b>The left arm.</b> This is the biggest one. In the original it holds a set position;
        at the moment it moves around a lot on its own.</li>
    <li><b>The feet stay put.</b> In the original his feet never leave their starting position —
        no lunge, no step through. Wide, planted, and they stay there.</li>
    <li><b>Start and finish in the same pose</b>, and there is a brief return to it mid-phrase too.</li>
  </ul>
</div>

<div class="film">
  <b>Two filming changes</b>
  <ul>
    <li><b>Shoot against the white wall, not the dark curtain.</b> Keep exactly the same
        kit — black against white is ideal, it is only black-on-black that causes problems.
        The short-sleeved tee in particular is a real help: your forearms stay clearly
        readable even at the bottom of the crouch, where everything folds together.
        The side view is the angle I need most.</li>
    <li><b>The 45° shot was closer to 20°.</b> Floor tape would help — mark the camera
        position and your foot positions so the setup repeats.</li>
  </ul>
  Everything else stays exactly as before: whole body in frame, stabilisation off, 60 fps,
  landscape, raw files. Don't worry about sound.
</div>

<div class="slow">
  <b>Please slow it right down — and don't worry about timing at all</b>
  <p style="margin:6pt 0 0">This is the change that will help most, and it should make your
  job easier rather than harder.</p>
  <p style="margin:6pt 0 0"><b>Per camera position, four passes:</b></p>
  <ul>
    <li><b>First two: as slow as you like.</b> Half speed, quarter speed, whatever lets you
        hit each position exactly. These are the important ones — I only need the
        <i>positions</i> to be right. <b>Ignore the timing completely.</b></li>
    <li><b>Then two at normal performance speed</b>, for the feel of it.</li>
  </ul>
  <p style="margin:8pt 0 0">I can fix the timing afterwards — I match your frames up against
  the original one by one, so I can stretch or compress any part of it later. What I
  <i>can't</i> fix afterwards is a position that isn't quite right. So accuracy is worth far
  more to me than speed.</p>
  <p style="margin:8pt 0 0">For reference, comparing your take against the original beat by
  beat: overall you're running about <b>1.3&times; faster</b>, but unevenly. The uppercut half
  is consistently <b>1.5&ndash;2&times; too quick</b>, and the release out of the throw is nearly
  <b>6&times; too quick</b> — while a couple of the early beats are actually slower than the
  original. That unevenness is much easier to get right slowly.</p>
</div>

<div class="pagebreak"></div>
<h2>Beat by beat</h2>
{CARDS}

<p class="foot">Original footage: Mortal Kombat behind-the-scenes, 1992 — used here as movement
reference only.</p>
</body></html>"""

tmp_html = "/tmp/feedback.html"
open(tmp_html, "w").write(HTML)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
                "--no-pdf-header-footer", f"--print-to-pdf={OUT}", tmp_html],
               check=True, capture_output=True)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
