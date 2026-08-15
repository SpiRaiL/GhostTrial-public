#!/usr/bin/env python
"""Isolate the Scorpion actor from the MK behind-the-scenes frames.

The clip is a locked-off camera on the digitising stage, but the actor never
steps off his mark during the whole 100 s take, so a temporal-median plate is
contaminated with a ghost of him and background subtraction is useless here.
We segment each frame instead (rembg / isnet-general-use) and clean the matte
with a largest-connected-component + hole-fill pass.

Writes, for frames_full/frame_%04d.jpg:
  masks/            8-bit matte, inner-frame geometry
  character_alpha/  RGBA cutout, inner-frame geometry (absolute position kept)
  character_tight/  RGBA cutout cropped to the per-frame bounding box
  character_boxed/  JPEG crop, background left intact
  bboxes.csv        per-frame bbox in ORIGINAL 1280x720 frame coordinates
"""
import csv
import glob
import os

import cv2
import numpy as np
from rembg import new_session, remove

# Inner video rect inside the arcade-cabinet border composite (1280x720 source).
X0, X1, Y0, Y1 = 172, 1110, 10, 712
FPS = 30000 / 1001
CLIP_START_S = 120.0  # frames_full starts at 2:00 of the source video
# The matte is confident on the bright costume but weak on the black trousers,
# which sit against a dark floor -- a single threshold at 128 severs the legs.
# Hysteresis instead: grow from confident cores out through the weak region, and
# drop any weak blob with no core in it (that is the cast shadow on the wall).
CORE_T, LOOSE_T = 140, 45
CORE_MIN, MIN_AREA = 200, 300
MAX_HOLE = 2000  # px; larger enclosed gaps are real background, not pinholes
ALPHA_THRESH = 128
PAD = 12  # px of breathing room around the tight crop

HERE = os.path.dirname(os.path.abspath(__file__))


def clean_mask(alpha):
    """Hysteresis-binarise, drop specks and shadow, close pinholes, feather."""
    core = alpha > CORE_T
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > LOOSE_T).astype(np.uint8), 8)
    m = np.zeros(alpha.shape, np.uint8)
    for i in range(1, n):
        blob = lab == i
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA and core[blob].sum() >= CORE_MIN:
            m[blob] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # Fill enclosed pinholes only. Filling every hole swallows the gap between
    # the legs on frames where the contact shadow bridges the two boots.
    nh, hlab, hstats, _ = cv2.connectedComponentsWithStats(1 - m, 8)
    for i in range(1, nh):
        if hstats[i, cv2.CC_STAT_AREA] > MAX_HOLE:
            continue
        x, y = hstats[i, cv2.CC_STAT_LEFT], hstats[i, cv2.CC_STAT_TOP]
        bw, bh = hstats[i, cv2.CC_STAT_WIDTH], hstats[i, cv2.CC_STAT_HEIGHT]
        if x == 0 or y == 0 or x + bw == m.shape[1] or y + bh == m.shape[0]:
            continue  # touches the border, so it is background, not a hole
        m[hlab == i] = 1
    soft = cv2.GaussianBlur(m * 255, (5, 5), 0)
    return np.where(m > 0, np.maximum(soft, 200), soft).astype(np.uint8)


def main():
    for d in ("masks", "character_alpha", "character_tight", "character_boxed"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    frames = sorted(glob.glob(os.path.join(HERE, "frames_full", "*.jpg")))
    session = new_session("isnet-general-use")
    rows = []

    for i, path in enumerate(frames, start=1):
        inner = cv2.imread(path)[Y0:Y1, X0:X1]
        cut = remove(cv2.cvtColor(inner, cv2.COLOR_BGR2RGB), session=session)
        mask = clean_mask(np.array(cut)[:, :, 3])

        ys, xs = np.where(mask > ALPHA_THRESH)
        x, y, w, h = xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1

        rgba = np.dstack([inner, mask])
        name = f"frame_{i:04d}"
        cv2.imwrite(os.path.join(HERE, "masks", name + ".png"), mask)
        cv2.imwrite(os.path.join(HERE, "character_alpha", name + ".png"), rgba)

        cx0, cy0 = max(0, x - PAD), max(0, y - PAD)
        cx1, cy1 = min(inner.shape[1], x + w + PAD), min(inner.shape[0], y + h + PAD)
        cv2.imwrite(os.path.join(HERE, "character_tight", name + ".png"),
                    rgba[cy0:cy1, cx0:cx1])
        cv2.imwrite(os.path.join(HERE, "character_boxed", name + ".jpg"),
                    inner[cy0:cy1, cx0:cx1], [cv2.IMWRITE_JPEG_QUALITY, 95])

        rows.append({
            "frame": i,
            "source_time_s": round(CLIP_START_S + (i - 1) / FPS, 4),
            # bbox expressed in the original 1280x720 frame
            "x": int(x) + X0, "y": int(y) + Y0, "w": int(w), "h": int(h),
            "area_px": int((mask > ALPHA_THRESH).sum()),
        })
        if i % 30 == 0:
            print(f"  {i}/{len(frames)}")

    with open(os.path.join(HERE, "bboxes.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"done: {len(rows)} frames")


if __name__ == "__main__":
    main()
