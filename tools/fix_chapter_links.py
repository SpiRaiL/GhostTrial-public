#!/usr/bin/env python3
"""Make chapters linkable, and make a link actually open the chapter it points at.

    .venv/bin/python tools/fix_chapter_links.py reports/*.html

Two faults, both from the collapsible rewrite.

First, the split reports carry NO ids at all — the chapter <details> elements were
written without them, so there was never anything for a link to target. Every
chapter gets a stable slug derived from its title, so #ch-where-the-error-actually-is
keeps working when chapters are renumbered or reordered.

Second, an anchor into a COLLAPSED <details> lands on nothing: the browser scrolls
to an element of zero height, whose media is still parked in data-src. So the JS
here opens the target, hydrates it, and only then scrolls — on first load, on every
hashchange, and on in-page clicks.

Each chapter also gets a # handle beside its title that copies its own deep link,
so a specific result can be pointed at directly rather than "open report 8 and
scroll".
"""

import re
import sys
import unicodedata

CSS = """
<style id="chlink-css">
.chap-anchor { margin-left:auto; color:#3a4854; text-decoration:none; font-size:13px;
  padding:0 4px; flex:none; }
details.chap > summary:hover .chap-anchor { color:#59b0ff; }
.chap-anchor.copied { color:#4ec9a5; }
details.chap:target { border-color:#59b0ff; }
/* Media carries its real pixel size as width/height attributes (see the stamping
   step in the commit that added this), so the browser reserves the correct box
   before the file loads. Without it an anchor scrolls to a position that the media
   then pushes down, and the reader lands in a blank gap. height:auto keeps it
   responsive; aspect-ratio covers anything that was missed. */
figure video, figure img { height:auto; aspect-ratio:auto 16/9; }
</style>
"""

JS = """
<script id="chlink-js">
// An anchor into a collapsed <details> lands on a zero-height element whose media
// is still parked in data-src. Open it, swap the media in, then scroll.
function chHydrate(d){
  d.querySelectorAll('[data-src]').forEach(function(el){
    el.src = el.getAttribute('data-src'); el.removeAttribute('data-src');
  });
}
function chReveal(hash, smooth){
  if (!hash || hash.length < 2) return false;
  var el = document.getElementById(hash.slice(1));
  if (!el) return false;
  var d = el.closest ? el.closest('details.chap') : null;
  if (!d && el.tagName === 'DETAILS') d = el;
  if (d){ d.open = true; chHydrate(d); }
  // let the open reflow before measuring where to scroll to, then correct again
  // once the chapter's media has its real height
  var go = function(sm){
    el.scrollIntoView({behavior: sm ? 'smooth' : 'auto', block: 'start'});
  };
  requestAnimationFrame(function(){ go(smooth); });
  if (d) d.querySelectorAll('video,img').forEach(function(m){
    m.addEventListener('loadedmetadata', function(){ go(false); }, {once:true});
    m.addEventListener('load', function(){ go(false); }, {once:true});
  });
  window.addEventListener('load', function(){ go(false); }, {once:true});
  return true;
}
document.querySelectorAll('details.chap').forEach(function(d){
  if (d.open) chHydrate(d);
  d.addEventListener('toggle', function(){ if (d.open) chHydrate(d); });
});
window.addEventListener('hashchange', function(){ chReveal(location.hash, true); });
document.addEventListener('click', function(e){
  var a = e.target.closest && e.target.closest('a[href^="#"]');
  if (!a) return;
  var h = a.getAttribute('href');
  if (h === '#') return;
  if (chReveal(h, true)){ e.preventDefault(); history.replaceState(null, '', h); }
});
// copy handle on each chapter
document.querySelectorAll('.chap-anchor').forEach(function(a){
  a.addEventListener('click', function(e){
    e.preventDefault(); e.stopPropagation();
    var url = location.href.split('#')[0] + a.getAttribute('href');
    if (navigator.clipboard) navigator.clipboard.writeText(url);
    history.replaceState(null, '', a.getAttribute('href'));
    a.classList.add('copied'); a.textContent = 'copied';
    setTimeout(function(){ a.classList.remove('copied'); a.textContent = '#'; }, 1200);
  });
});
chReveal(location.hash, false);
</script>
"""


def slug(text):
    t = re.sub(r"<[^>]+>", "", text)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return "ch-" + re.sub(r"-+", "-", t)[:60]


def fix(path):
    s = open(path).read()
    out, seen, n = s, set(), 0

    # id and handle in one pass: the slug comes from the chapter's own title, which
    # lives in the <summary> that follows the tag being edited
    def add_id(m):
        nonlocal n
        attrs, summary = m.group(1), m.group(2)
        sid = re.search(r'id="([^"]+)"', attrs)
        if sid:
            sid = sid.group(1)
            attrs_out = attrs
        else:
            ttl = re.search(r'class="chttl"[^>]*>(.*?)</span>', summary, re.S)
            sid = slug(ttl.group(1) if ttl else summary)
            while sid in seen:
                sid += "-b"
            attrs_out = f'{attrs} id="{sid}"'
            n += 1
        seen.add(sid)
        if "chap-anchor" in summary:
            return f"<details{attrs_out}><summary>{summary}</summary>"
        handle = (f'<a class="chap-anchor" href="#{sid}" '
                  f'title="link to this chapter">#</a>')
        return f"<details{attrs_out}><summary>{summary}{handle}</summary>"

    out = re.sub(r'<details((?:(?!>).)*?class="chap"(?:(?!>).)*?)>\s*<summary>(.*?)</summary>',
                 add_id, out, flags=re.S)

    # one copy of the behaviour, replacing any earlier version
    out = re.sub(r'<style id="chlink-css">.*?</style>', "", out, flags=re.S)
    out = re.sub(r'<script id="chlink-js">.*?</script>', "", out, flags=re.S)
    out = out.replace("</head>", CSS + "</head>", 1)
    out = out.replace("</body>", JS + "</body>", 1)
    open(path, "w").write(out)
    return n


for p in sys.argv[1:]:
    print(f"{p.split('/')[-1]:24s} {fix(p):2d} chapters made linkable")
