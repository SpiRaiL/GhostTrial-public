#!/usr/bin/env python3
"""Manual frame-pairing UI: reference on the left, actor on the right.

    .venv/bin/python tools/sync_ui.py            # then open the tailnet URL it prints

Automatic phase matching kept mis-picking beats (it grabbed the coil instead of the
rise, and mis-timed the sink), so the alignment is done by hand instead. Pick a frame
on each side, type a comment, save. Pairs land in data/sync_pairs.json and can be
turned into a comparison sequence afterwards.

Serves proxies from data/sync_media (H.264, 720p, all-keyframe so stepping is exact).
Range requests are implemented because seeking needs them.
"""

import json
import os
import re
import subprocess
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(REPO, "data", "sync_media")
PAIRS = os.path.join(REPO, "data", "sync_pairs.json")
PORT = 8765


def clips():
    out = []
    for fn in sorted(os.listdir(MEDIA)):
        if not fn.endswith(".mp4"):
            continue
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate,nb_frames",
                            "-show_entries", "format=duration", "-of", "json",
                            os.path.join(MEDIA, fn)], capture_output=True, text=True)
        d = json.loads(r.stdout)
        rate = d["streams"][0]["r_frame_rate"]
        n, _, den = rate.partition("/")
        fps = float(n) / float(den or 1)
        out.append(dict(file=fn, fps=round(fps, 4),
                        dur=float(d["format"]["duration"]),
                        side="ref" if fn.startswith("ref") else "actor",
                        label=fn[:-4].replace("_", " ")))
    return out


def load_pairs():
    if os.path.exists(PAIRS):
        return json.load(open(PAIRS))
    return []


def save_pairs(p):
    os.makedirs(os.path.dirname(PAIRS), exist_ok=True)
    json.dump(p, open(PAIRS, "w"), indent=2)


HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Frame sync — Ghost Trial 03</title><style>
:root{--bg:#0f1216;--ink:#e7edf3;--dim:#9aa7b4;--line:#2a333d;--acc:#59b0ff;--good:#4ec9a3;--warn:#e6b34d;--bad:#e0736b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1700px;margin:0 auto;padding:14px 18px 60px}
h1{font-size:19px;margin:0 0 2px}.sub{color:var(--dim);font-size:13px;margin:0 0 14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{border:1px solid var(--line);border-radius:10px;padding:10px;background:#141a21}
video{width:100%;background:#000;border-radius:6px;display:block}
select,input,button,textarea{font:inherit;background:#0b0e12;color:var(--ink);
border:1px solid var(--line);border-radius:6px;padding:7px 9px}
button{cursor:pointer}button:hover{border-color:var(--acc)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.tag{background:#0b0e12;border:1px solid var(--line);border-radius:6px;padding:5px 9px}
.big{font-size:17px;font-weight:700;color:var(--acc)}
.save{background:#12351f;border-color:var(--good);color:#dff5ea;font-weight:700;padding:10px 20px}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13.5px}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--dim);font-size:12px;text-transform:uppercase}
.del{color:var(--bad);cursor:pointer;border:none;background:none;font-size:16px}
kbd{background:#0b0e12;border:1px solid var(--line);border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body><div class=wrap>
<h1>Frame sync — reference vs actor</h1>
<p class=sub>Line up the same beat on both sides, add a comment, save.
<kbd>A</kbd>/<kbd>D</kbd> step reference &nbsp; <kbd>J</kbd>/<kbd>L</kbd> step actor &nbsp;
hold <kbd>Shift</kbd> for ×10 &nbsp; <kbd>Enter</kbd> saves</p>
<div class=grid>
  <div class=panel>
    <select id=selref style=width:100%></select>
    <video id=vref preload=auto></video>
    <div class=row>
      <button onclick="step('ref',-10)">◀◀</button><button onclick="step('ref',-1)">◀</button>
      <button onclick="step('ref',1)">▶</button><button onclick="step('ref',10)">▶▶</button>
      <button onclick="play('ref')">play</button>
      <span class="tag mono" id=infref>–</span>
    </div>
    <input type=range id=barref min=0 max=1000 value=0 style=width:100%;margin-top:8px>
  </div>
  <div class=panel>
    <select id=selact style=width:100%></select>
    <video id=vact preload=auto></video>
    <div class=row>
      <button onclick="step('act',-10)">◀◀</button><button onclick="step('act',-1)">◀</button>
      <button onclick="step('act',1)">▶</button><button onclick="step('act',10)">▶▶</button>
      <button onclick="play('act')">play</button>
      <span class="tag mono" id=infact>–</span>
    </div>
    <input type=range id=baract min=0 max=1000 value=0 style=width:100%;margin-top:8px>
  </div>
</div>
<div class=row style=margin-top:14px>
  <input id=label placeholder="beat name (e.g. deepest crouch)" style=width:260px>
  <input id=comment placeholder="comment — what matches, what to change" style=flex:1;min-width:340px>
  <button class=save onclick=savePair()>Save pair</button>
  <span id=status class=mono style=color:var(--good)></span>
</div>
<table id=tbl><thead><tr><th>#</th><th>beat</th><th>reference</th><th>actor</th><th>comment</th><th></th></tr></thead><tbody></tbody></table>
<script>
let CLIPS=[],P=[];
const V={ref:document.getElementById('vref'),act:document.getElementById('vact')};
const S={ref:document.getElementById('selref'),act:document.getElementById('selact')};
const B={ref:document.getElementById('barref'),act:document.getElementById('baract')};
const N={ref:document.getElementById('infref'),act:document.getElementById('infact')};
function cur(w){return CLIPS.find(c=>c.file===S[w].value)}
function fps(w){const c=cur(w);return c?c.fps:30}
function upd(w){const c=cur(w);if(!c)return;const t=V[w].currentTime;
  N[w].textContent=`frame ${Math.round(t*c.fps)}  ·  ${t.toFixed(3)}s`;
  B[w].value=Math.round(1000*t/c.dur);}
function step(w,n){V[w].pause();V[w].currentTime=Math.max(0,V[w].currentTime+n/fps(w));}
function play(w){V[w].paused?V[w].play():V[w].pause();}
for(const w of ['ref','act']){
  V[w].addEventListener('timeupdate',()=>upd(w));
  V[w].addEventListener('seeked',()=>upd(w));
  V[w].addEventListener('loadedmetadata',()=>upd(w));
  B[w].addEventListener('input',()=>{const c=cur(w);V[w].pause();V[w].currentTime=c.dur*B[w].value/1000;});
  S[w].addEventListener('change',()=>{V[w].src='/media/'+S[w].value;V[w].load();});
}
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA'].includes(e.target.tagName)){if(e.key==='Enter')savePair();return;}
  const m=e.shiftKey?10:1;
  if(e.key==='a'||e.key==='A')step('ref',-m); if(e.key==='d'||e.key==='D')step('ref',m);
  if(e.key==='j'||e.key==='J')step('act',-m); if(e.key==='l'||e.key==='L')step('act',m);
  if(e.key==='Enter')savePair();
});
function render(){
  const tb=document.querySelector('#tbl tbody');tb.innerHTML='';
  P.forEach((p,i)=>{const tr=document.createElement('tr');
    tr.innerHTML=`<td class=mono>${i+1}</td><td>${p.label||''}</td>
    <td class=mono>${p.ref_clip}<br>f${p.ref_frame} · ${p.ref_t.toFixed(3)}s</td>
    <td class=mono>${p.act_clip}<br>f${p.act_frame} · ${p.act_t.toFixed(3)}s</td>
    <td>${p.comment||''}</td>
    <td><button class=del onclick=del(${i})>✕</button></td>`;tb.appendChild(tr);});
}
async function savePair(){
  const rc=cur('ref'),ac=cur('act');if(!rc||!ac)return;
  const p={label:document.getElementById('label').value,
    comment:document.getElementById('comment').value,
    ref_clip:rc.file,ref_t:V.ref.currentTime,ref_frame:Math.round(V.ref.currentTime*rc.fps),ref_fps:rc.fps,
    act_clip:ac.file,act_t:V.act.currentTime,act_frame:Math.round(V.act.currentTime*ac.fps),act_fps:ac.fps};
  P.push(p);await fetch('/save',{method:'POST',body:JSON.stringify(P)});
  document.getElementById('comment').value='';document.getElementById('label').value='';
  document.getElementById('status').textContent='saved '+P.length;render();
}
async function del(i){P.splice(i,1);await fetch('/save',{method:'POST',body:JSON.stringify(P)});render();}
(async()=>{
  CLIPS=await (await fetch('/clips')).json();
  P=await (await fetch('/pairs')).json();
  for(const w of ['ref','act']){
    const want=w==='ref'?'ref':'actor';
    CLIPS.filter(c=>c.side===want).forEach(c=>{
      const o=document.createElement('option');o.value=c.file;o.textContent=c.label+`  (${c.fps} fps)`;S[w].appendChild(o);});
    // default the actor panel to Side view - the fight-relative camera, the one being re-shot
    if(w==='act'){for(const o of S[w].options){if(o.value.includes('side')){S[w].value=o.value;break;}}}
    if(S[w].options.length){V[w].src='/media/'+S[w].value;}
  }
  render();
})();
</script></div></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def handle_one_request(self):
        # the browser aborts range requests constantly while seeking; that is normal
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_GET(self):
        if self.path == "/":
            return self._send(200, HTML, "text/html; charset=utf-8")
        if self.path == "/clips":
            return self._send(200, json.dumps(clips()))
        if self.path == "/pairs":
            return self._send(200, json.dumps(load_pairs()))
        if self.path.startswith("/media/"):
            fn = os.path.basename(self.path[7:])
            path = os.path.join(MEDIA, fn)
            if not os.path.exists(path):
                return self._send(404, b"no")
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            if rng:                                  # seeking needs byte ranges
                m = re.match(r"bytes=(\d+)-(\d*)", rng)
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else size - 1
                b = min(b, size - 1)
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                self.send_header("Content-Length", str(b - a + 1))
                self.end_headers()
                with open(path, "rb") as fh:
                    fh.seek(a)
                    self.wfile.write(fh.read(b - a + 1))
            else:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with open(path, "rb") as fh:
                    self.wfile.write(fh.read())
            return
        self._send(404, b"no")

    def do_POST(self):
        if self.path == "/save":
            n = int(self.headers.get("Content-Length", 0))
            save_pairs(json.loads(self.rfile.read(n) or b"[]"))
            return self._send(200, b'{"ok":true}')
        self._send(404, b"no")


if __name__ == "__main__":
    ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                        text=True).stdout.strip().split("\n")[0] or \
        socket.gethostbyname(socket.gethostname())
    print(f"clips: {len(clips())}   pairs so far: {len(load_pairs())}")
    print(f"\n  http://{ip}:{PORT}/\n")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
