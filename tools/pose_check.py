#!/usr/bin/env python3
"""Run 2D pose over delivered capture footage and check it against the brief.

    .venv/bin/python tools/pose_check.py raw/capture-01 --out data/capture_analysis

Background subtraction is useless on this footage — the studio curtain drifts in
the air and lights up the whole mask (see /tmp/mask_Side_view.png). So detect the
performer properly, with torchvision's COCO keypoint R-CNN on the GPU.

Per frame it records the 17 COCO keypoints, from which the checks that matter fall
out directly:

  framing      are the ankles and the head inside the frame, with margin?
  subject size how much of the frame height does the body actually occupy?
  takes        wrist speed segments the reps
  the move     hip height gives crouch depth; wrist-above-head gives the rise;
               wrist-ahead-of-shoulder gives the throw extension and its hold
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch
from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                          keypointrcnn_resnet50_fpn)

# COCO keypoint order
KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb",
      "r_elb", "l_wri", "r_wri", "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
I = {n: i for i, n in enumerate(KP)}

W, H = 640, 360          # detection resolution
STRIDE = 3               # every 3rd frame ~ 20 Hz, plenty for 60 fps footage


def frames(path, w=W, h=H, stride=1):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-vf", f"scale={w}:{h}", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    n = w * h * 3
    i = 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        if i % stride == 0:
            yield i, np.frombuffer(b, np.uint8).reshape(h, w, 3)
        i += 1
    p.stdout.close()
    p.wait()


def run(path, model, dev, batch=12):
    out = []
    buf, idx = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            preds = model([t.to(dev) for t in buf])
        for j, pr in zip(idx, preds):
            if len(pr["scores"]) == 0 or float(pr["scores"][0]) < 0.85:
                out.append((j, *([np.nan] * 34), 0.0))
                continue
            k = pr["keypoints"][0].cpu().numpy()      # (17,3)
            out.append((j, *k[:, 0].tolist(), *k[:, 1].tolist(),
                        float(pr["scores"][0])))
        buf.clear()
        idx.clear()

    for i, f in frames(path, stride=STRIDE):
        buf.append(torch.from_numpy(f.copy()).permute(2, 0, 1).float() / 255.0)
        idx.append(i)
        if len(buf) >= batch:
            flush()
    flush()
    return np.array(out, dtype=np.float64)


def report(name, a, fps):
    ok = a[:, -1] > 0
    x = a[:, 1:18]
    y = a[:, 18:35]
    d = {"frames_scored": int(ok.sum()), "frames_total": int(len(a))}

    ank = np.nanmin(np.stack([y[:, I["l_ank"]], y[:, I["r_ank"]]]), axis=0)
    ankmax = np.nanmax(np.stack([y[:, I["l_ank"]], y[:, I["r_ank"]]]), axis=0)
    head = y[:, I["nose"]]
    hip = np.nanmean(np.stack([y[:, I["l_hip"]], y[:, I["r_hip"]]]), axis=0)

    d["ankle_lowest_px"] = round(float(np.nanmax(ankmax)), 1)
    d["ankle_margin_px"] = round(float(H - np.nanmax(ankmax)), 1)
    d["head_highest_px"] = round(float(np.nanmin(head)), 1)
    d["frames_ankle_within_10px_of_bottom"] = int(np.nansum(ankmax > H - 10))
    d["frames_head_within_10px_of_top"] = int(np.nansum(head < 10))

    body_h = ankmax - head
    d["subject_height_pct"] = [round(float(np.nanpercentile(body_h / H * 100, p)), 1)
                               for p in (5, 50, 95)]

    # crouch depth: hip height above the lowest ankle, normalised by standing value
    hip_above = ankmax - hip
    stand = np.nanpercentile(hip_above, 90)
    d["hip_drop_pct_of_standing"] = round(float(100 * (1 - np.nanmin(hip_above) / stand)), 1)

    # arm work, in body-height units
    scale = np.nanmedian(body_h)
    reach = np.nanmax(np.stack([
        np.abs(x[:, I["r_wri"]] - x[:, I["r_sho"]]),
        np.abs(x[:, I["l_wri"]] - x[:, I["l_sho"]])]), axis=0) / scale
    d["max_horizontal_reach_bodyheights"] = round(float(np.nanmax(reach)), 3)
    above = np.nanmin(np.stack([y[:, I["r_wri"]], y[:, I["l_wri"]]]), axis=0)
    d["wrist_above_head_px"] = round(float(np.nanmin(head - above)), 1)

    # takes, from wrist speed
    wr = np.stack([x[:, I["r_wri"]], y[:, I["r_wri"]]], axis=1)
    v = np.r_[0, np.linalg.norm(np.diff(wr, axis=0), axis=1)]
    v = np.nan_to_num(v)
    thr = np.nanpercentile(v, 75)
    act = np.convolve((v > thr).astype(float), np.ones(9) / 9, "same") > 0.25
    takes, s, gap = [], None, 0
    step = STRIDE / fps
    for i, aa in enumerate(act):
        if aa:
            s = i if s is None else s
            gap = 0
        elif s is not None:
            gap += 1
            if gap * step > 0.7:
                if (i - gap - s) * step > 1.5:
                    takes.append((round(s * step, 2), round((i - gap) * step, 2)))
                s = None
                gap = 0
    if s is not None:
        takes.append((round(s * step, 2), round(len(act) * step, 2)))
    d["takes"] = takes
    d["n_takes"] = len(takes)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--out", default="data/capture_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(dev)

    summary = {}
    for fn in sorted(os.listdir(args.indir)):
        if not fn.lower().endswith((".mov", ".mp4")):
            continue
        path = os.path.join(args.indir, fn)
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", path],
            capture_output=True, text=True).stdout.strip().strip(",")
        num, _, den = raw.partition("/")
        fps = float(num) / float(den or 1)
        print(f"\n=== {fn}", flush=True)
        a = run(path, model, dev)
        np.save(os.path.join(args.out, fn.rsplit(".", 1)[0] + "_kp.npy"), a)
        d = report(fn, a, fps)
        summary[fn] = d
        for k, v in d.items():
            if k != "takes":
                print(f"   {k}: {v}")
        print(f"   takes: {d['takes']}")

    with open(os.path.join(args.out, "pose_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out}/pose_summary.json")


if __name__ == "__main__":
    main()
