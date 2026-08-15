#!/usr/bin/env python3
"""Materialise Git-LFS pointer files without git-lfs installed.

    .venv/bin/python tools/fetch_lfs.py vendor/GR00T-WholeBodyControl NVlabs/GR00T-WholeBodyControl main

`git clone --depth 1` leaves LFS-tracked files as ~130-byte pointers. In this repo
that includes every `*.STL` mesh the G1 URDF references, which makes Isaac Lab's
URDF->USD conversion die with:

    Failed to open layer @/tmp/IsaacLab/usd_*/configuration/pelvis.tmp.usd@

GitHub serves LFS content for public repos from media.githubusercontent.com, so the
pointers can be resolved with plain HTTP and no git-lfs binary.
"""

import concurrent.futures as cf
import os
import sys
import urllib.request

POINTER = b"version https://git-lfs.github.com/spec/v1"
ROOT, REPO, REF = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "main")


def is_pointer(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(POINTER))
        return head == POINTER
    except OSError:
        return False


def declared_size(path):
    with open(path) as fh:
        for line in fh:
            if line.startswith("size "):
                return int(line.split()[1])
    return None


def fetch(rel):
    dst = os.path.join(ROOT, rel)
    want = declared_size(dst)
    url = f"https://media.githubusercontent.com/media/{REPO}/{REF}/{rel}"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except Exception as e:
        return rel, f"FAIL {type(e).__name__}"
    if want is not None and len(data) != want:
        return rel, f"SIZE MISMATCH got {len(data)} want {want}"
    with open(dst, "wb") as fh:
        fh.write(data)
    return rel, f"ok {len(data)}"


pointers = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != ".git"]
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        if is_pointer(p):
            pointers.append(os.path.relpath(p, ROOT))

print(f"{len(pointers)} LFS pointers under {ROOT}")
bad = 0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for rel, status in ex.map(fetch, pointers):
        if not status.startswith("ok"):
            bad += 1
            print(f"  {status}: {rel}")
print(f"done — {len(pointers) - bad} fetched, {bad} failed")
