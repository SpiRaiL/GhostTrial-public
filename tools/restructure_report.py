#!/usr/bin/env python3
"""Put the report in chronological order, and make chapters collapse and load lazily.

    .venv/bin/python tools/restructure_report.py "<report>.html"

Two problems this fixes.

Order: new findings kept getting inserted ABOVE older ones, because each edit
anchored on whatever text was convenient. The result reads backwards in places —
the static-feasibility finding sits before the training run that motivated it. The
chapter order is therefore declared explicitly here, in the order the work actually
happened, and applied whatever order the file is currently in.

Weight: the page carries 9 videos and 21 images, and a browser fetches all of them
on load. Every chapter becomes a <details> and its media is parked in data-src
until that chapter is opened, so opening the page costs one chapter, not thirty
assets. It stays a single self-contained .html — rc-doc serves plain files that
must also work opened locally, so this cannot depend on the server.
"""

import os
import re
import sys

PATH = sys.argv[1]
html = open(PATH).read()

# the order the work happened in
ORDER = [
    "video", "pipeline", "verify", "floating", "phases", "risks", "bones", "plan",
    "repro", "g1direct", "training", "capture", "human", "takes",
    "trainrun", "beforeafter", "a5state", "static", "balanced",
    "hardware", "files",
]
TITLES = {
    "video": "What it looks like",
    "pipeline": "The pipeline, and why it was cheap",
    "verify": "Validation, not vibes",
    "floating": "The clip was floating",
    "phases": "What the robot actually does",
    "risks": "Risks, in order",
    "bones": "BONES-SEED: 49 GB, and not the move we wanted",
    "plan": "Plan — for whoever picks this up",
    "repro": "Reproduce",
    "g1direct": "Author in the robot's joint space",
    "training": "The training loop runs — locally, for nothing",
    "capture": "The capture came back — reviewed against the brief",
    "human": "The human first — a .blend of the motion before the robot",
    "takes": "The take we will train on — A2, then A3",
    "trainrun": "The fine-tune — two runs, and what the 2× swing costs",
    "beforeafter": "Before and after, against the stock baseline",
    "a5state": "A5 under physics — it stays up, it does not do the move",
    "static": "Why none of it worked: the reference cannot be stood in",
    "balanced": "Authoring balanced poses — the physics track",
    "hardware": "Hardware access — draft to WeBot",
    "files": "Where everything is",
}

head, body = html.split("<h2 ", 1)
body = "<h2 " + body
tail = ""
m = re.search(r'(<p class="foot".*)$', body, re.S)
if m:
    tail = m.group(1)
    body = body[:m.start()]

# split into chapters on <h2 id="...">
parts = re.split(r'(?=<h2 id=")', body)
chapters = {}
for p in parts:
    mm = re.match(r'<h2 id="([^"]+)"', p)
    if mm:
        chapters[mm.group(1)] = p

# promote the three h3s that were wrongly nested inside the fine-tune chapter
if "trainrun" in chapters:
    src = chapters["trainrun"]
    for cid, marker in (("beforeafter", "Before and after, against the stock baseline"),
                        ("a5state", "Where it actually stands — A5 under physics"),
                        ("static", "Why none of it was working"),
                        ("balanced", "Authoring balanced poses — the physics track")):
        i = src.find(f">{marker}")
        if i < 0:
            continue
        start = src.rfind("<h3", 0, i)
        nxt = [j for j in (src.find("<h3", i), len(src)) if j > 0]
        end = min(nxt)
        block = src[start:end]
        src = src[:start] + src[end:]
        body_only = re.sub(r"^<h3[^>]*>.*?</h3>", "", block, flags=re.S)
        chapters[cid] = f'<h2 id="{cid}">{TITLES[cid]}</h2>\n' + body_only
    chapters["trainrun"] = src

out = []
for n, cid in enumerate([c for c in ORDER if c in chapters], start=1):
    c = chapters[cid]
    c = re.sub(r'<h2 id="[^"]+">.*?</h2>',
               f'<h2 id="{cid}">{n} · {TITLES[cid]}</h2>', c, count=1, flags=re.S)
    # park media until the chapter is opened
    c = re.sub(r'(<video[^>]*?)\ssrc="([^"]+)"', r'\1 data-src="\2" preload="none"', c)
    c = re.sub(r'(<img[^>]*?)\ssrc="([^"]+)"', r'\1 data-src="\2" loading="lazy"', c)
    heading = re.search(r"<h2[^>]*>(.*?)</h2>", c, re.S).group(1)
    rest = re.sub(r"<h2[^>]*>.*?</h2>", "", c, count=1, flags=re.S)
    openattr = " open" if n <= 2 else ""
    out.append(f'<details class="chap" id="ch-{cid}"{openattr}>'
               f'<summary><span class="chnum">{n}</span>'
               f'<span class="chttl">{TITLES[cid]}</span></summary>\n{rest}</details>')

CSS = """
<style>
details.chap { border:1px solid #222c36; border-radius:8px; margin:10px 0; background:#0e1216; }
details.chap > summary { cursor:pointer; list-style:none; padding:13px 16px; display:flex;
  align-items:baseline; gap:12px; user-select:none; }
details.chap > summary::-webkit-details-marker { display:none; }
details.chap > summary::before { content:"▸"; color:#59b0ff; font-size:13px;
  transition:transform .12s; display:inline-block; }
details.chap[open] > summary::before { transform:rotate(90deg); }
details.chap > summary:hover { background:#141b22; }
.chnum { color:#59b0ff; font-family:ui-monospace,Menlo,monospace; font-size:13px; min-width:1.6em; }
.chttl { color:#e7edf3; font-weight:600; font-size:15px; }
details.chap > *:not(summary) { padding-left:16px; padding-right:16px; }
details.chap > *:last-child { padding-bottom:14px; }
.chaptools { margin:14px 0 4px; display:flex; gap:10px; }
.chaptools button { background:#16202a; color:#9aa7b4; border:1px solid #253039;
  border-radius:6px; padding:6px 12px; font-size:12px; cursor:pointer; }
.chaptools button:hover { color:#e7edf3; border-color:#3a4854; }
</style>
"""
JS = """
<script>
// Media is parked in data-src so opening the page costs one chapter, not every
// asset on it. Swap it in the first time a chapter is opened, then leave it.
function hydrate(d){
  d.querySelectorAll('[data-src]').forEach(function(el){
    el.src = el.getAttribute('data-src'); el.removeAttribute('data-src');
  });
}
document.querySelectorAll('details.chap').forEach(function(d){
  if (d.open) hydrate(d);
  d.addEventListener('toggle', function(){ if (d.open) hydrate(d); });
});
document.getElementById('expandAll').onclick = function(){
  document.querySelectorAll('details.chap').forEach(function(d){ d.open = true; hydrate(d); });
};
document.getElementById('collapseAll').onclick = function(){
  document.querySelectorAll('details.chap').forEach(function(d){ d.open = false; });
};
</script>
"""
TOOLS = ('<div class="chaptools"><button id="expandAll">Expand all</button>'
         '<button id="collapseAll">Collapse all</button></div>')

html_out = head.replace("</head>", CSS + "</head>") + TOOLS + "\n" + "\n".join(out) + tail
html_out = html_out.replace("</body>", JS + "</body>")
open(PATH, "w").write(html_out)
print(f"{len(out)} chapters, chronological, collapsible, media lazy-loaded")
for n, cid in enumerate([c for c in ORDER if c in chapters], start=1):
    print(f"  {n:2d}  {TITLES[cid]}")
