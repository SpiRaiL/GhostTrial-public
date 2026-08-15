#!/usr/bin/env python3
"""Draw a BVH as stick figures, to eyeball an authored pose without a GUI.

    .venv/bin/python tools/preview_bvh.py raw/authored/spear_throw.bvh /tmp/sheet.png

Uses GMR's own BVH loader, so what you see is exactly what the retargeter sees:
world-space joint positions, metres, Z-up.
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHAIN = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToeBase"),
]
RIGHT = {"RightShoulder", "RightArm", "RightForeArm", "RightHand",
         "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh")
    ap.add_argument("out_png")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--view", default="side", choices=["side", "front", "top"])
    ap.add_argument("--tile", type=int, default=260)
    args = ap.parse_args()

    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    frames, _ = load_bvh_file(args.bvh, format="nokov")

    picks = list(range(0, len(frames), args.every))
    S = args.tile
    sheet = Image.new("RGB", (S * min(len(picks), 8), S * ((len(picks) + 7) // 8)), (15, 18, 22))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)

    # common scale across all frames so poses are comparable
    allpts = np.array([[f[b][0] for b in f] for f in frames])
    zmin, zmax = allpts[..., 2].min(), allpts[..., 2].max()
    span = max(zmax - zmin, 1.2)

    for k, fi in enumerate(picks):
        fr = frames[fi]
        tile = Image.new("RGB", (S, S), (18, 22, 28))
        d = ImageDraw.Draw(tile)

        def proj(p):
            if args.view == "side":
                h = -p[1]          # -Y is forward, so forward is to the right
            elif args.view == "front":
                h = p[0]
            else:
                h = p[0]
            v = p[2] if args.view != "top" else -p[1]
            x = S / 2 + h / span * S * 0.8
            y = S - 24 - (v - zmin) / span * S * 0.8
            return x, y

        # ground line
        gy = S - 24 - (0 - zmin) / span * S * 0.8
        d.line([(0, gy), (S, gy)], fill=(60, 70, 82), width=1)

        for a, b in CHAIN:
            if a not in fr or b not in fr:
                continue
            col = (224, 115, 107) if b in RIGHT else (89, 176, 255)
            if b in ("Spine", "Spine1", "Spine2", "Neck", "Head"):
                col = (231, 237, 243)
            d.line([proj(fr[a][0]), proj(fr[b][0])], fill=col, width=3)
        for b in fr:
            x, y = proj(fr[b][0])
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(150, 165, 180))

        d.text((8, 6), f"f{fi}", fill=(154, 167, 180), font=font)
        sheet.paste(tile, (S * (k % 8), S * (k // 8)))

    sheet.save(args.out_png)
    hz = allpts[..., 2]
    print(f"{len(frames)} frames · z {zmin:.3f}..{zmax:.3f} m")
    head = np.array([f["Head"][0][2] for f in frames])
    foot = np.array([min(f["LeftToeBase"][0][2], f["RightToeBase"][0][2]) for f in frames])
    print(f"head z {head.min():.3f}..{head.max():.3f}   lowest toe z {foot.min():.3f}..{foot.max():.3f}")
    print(f"stature (head-foot) {(head-foot).min():.3f}..{(head-foot).max():.3f} m")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
