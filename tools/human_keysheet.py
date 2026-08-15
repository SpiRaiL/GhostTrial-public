#!/usr/bin/env python3
"""Contact sheet of the human motion at its own key beats.

    .venv/bin/python tools/human_keysheet.py <rendered-png-dir> <out.png>

The beats are found from the motion rather than picked by eye — pelvis height for
the crouch, hand reach for the throw, hand height for the uppercut — so the sheet
lands on the same moments the phrase is actually built from.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_human_bvh import fk, parse_bvh  # noqa: E402

PNGS, OUT = sys.argv[1], sys.argv[2]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the edited motion, to match what make_human_blend.py put in the .blend and
# therefore what the rendered frames actually show
BVH = os.path.join(REPO, "data", "human_bvh_edited",
                   os.environ.get("GT_TAKE", "A3_arm_clear_human.bvh"))

names, off, par, ch, data, T, fps = parse_bvh(BVH)
idx = {n: i for i, n in enumerate(names)}
pos = fk(names, off, par, ch, data)

feet = [idx[n] for n in ("LeftToeBase", "RightToeBase")]
floor = float(np.percentile(pos[:, feet, 1].min(axis=1), 2))
pelvis = pos[:, idx["Hips"], 1] - floor
hips_xz = pos[:, idx["Hips"]][:, [0, 2]]
lift = {h: pos[:, idx[h], 1] - floor for h in ("LeftHand", "RightHand")}


def arm_out(side):
    """How far the hand reaches horizontally from its own shoulder, 0..1 of arm length.

    Measured from the shoulder rather than the hips on purpose: through the throw
    he pitches his torso well forward, which puts both hands a long way from the
    hips even when they are tucked against his chest.
    """
    n = (np.linalg.norm(off[idx[side + "ForeArm"]])
         + np.linalg.norm(off[idx[side + "Hand"]])) * 0.01
    d = pos[:, idx[side + "Hand"]] - pos[:, idx[side + "Arm"]]
    return np.linalg.norm(d[:, [0, 2]], axis=1) / n


reach = {"LeftHand": arm_out("Left"), "RightHand": arm_out("Right")}


def peak(sig, a, b, lo=False):
    a, b = max(0, a), min(T, max(a + 1, b))
    return a + int(np.argmin(sig[a:b]) if lo else np.argmax(sig[a:b]))


# The phrase is spear throw FIRST, then uppercut — the crouch belongs to the
# uppercut half. An earlier version searched for the throw after the crouch and
# so labelled the uppercut's own drive as the throw; the two landmarks bracket
# the phrase and everything else is found relative to them.
f_crouch = int(np.argmin(pelvis))
# The throw is the ASYMMETRIC moment — one arm thrust out while the other stays
# tucked. Taking the furthest-reaching hand instead lands on the two-handed pull
# that follows, where both arms are just as extended.
gap = reach["RightHand"] - reach["LeftHand"]
f_throw = int(np.argmax(np.abs(gap[:f_crouch])))
throw_hand = "RightHand" if gap[f_throw] > 0 else "LeftHand"
punch_hand = max(lift, key=lambda h: lift[h][f_crouch:].max())
f_punch = peak(lift[punch_hand], f_crouch, T)
f_windup = peak(lift[throw_hand], 0, f_throw)          # arm cocked, before the thrust
# stillest frame between the throw's recovery and the crouch — the guard he resets to
speed = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(axis=1)
speed = np.convolve(speed, np.ones(15) / 15, mode="same")
f_guard = peak(speed, f_throw + 60, f_crouch - 30, lo=True)
f_reset = peak(speed, f_punch + 40, T - 1, lo=True)

BEATS = sorted({
    (peak(speed, 0, max(20, f_windup - 20), lo=True), "idle"),
    (f_windup, "wind-up"),
    (f_throw, f"SPEAR THROW · arm {reach[throw_hand][f_throw]*100:.0f}% extended"),
    ((f_throw + f_guard) // 2, "pull back"),
    (f_guard, "guard"),
    (f_crouch, f"crouch · pelvis {pelvis[f_crouch]:.2f} m"),
    ((f_crouch + f_punch) // 2, "drive"),
    (f_punch, f"UPPERCUT · hand {lift[punch_hand][f_punch]:.2f} m"),
    (f_reset, "reset"),
}, key=lambda b: b[0])

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
F_T, F_S, F_N, F_B = (ImageFont.truetype(FB, 30), ImageFont.truetype(FB, 16),
                      ImageFont.truetype(FB, 22), ImageFont.truetype(FR, 18))
BG, INK, DIM, ACC = (15, 18, 22), (231, 237, 243), (154, 167, 180), (89, 176, 255)

TH = 340
tiles = []
for f, lab in BEATS:
    im = Image.open(os.path.join(PNGS, f"f_{f + 1:04d}.png")).convert("RGB")
    im = im.crop((150, 40, 810, 700)).resize((TH, TH))
    tiles.append((f, lab, im))

COLS = 5
rows = (len(tiles) + COLS - 1) // COLS
CW, RHh = TH + 16, TH + 62
sheet = Image.new("RGB", (26 * 2 + CW * COLS, 116 + RHh * rows), BG)
d = ImageDraw.Draw(sheet)
d.text((26, 22), "The human motion, before any robot is involved", fill=INK, font=F_T)
d.text((26, 62), "the full combo — spear throw into uppercut · beats found from the motion itself · "
                 "78-joint SOMA skeleton solved from the phone footage", fill=DIM, font=F_B)
d.text((26, 88), f"throwing hand: {throw_hand}  ·  punching hand: {punch_hand}  ·  "
                 f"{T} frames at {fps:.0f} fps", fill=DIM, font=F_S)

for k, (f, lab, im) in enumerate(tiles):
    x, y = 26 + CW * (k % COLS), 116 + RHh * (k // COLS)
    sheet.paste(im, (x, y))
    d.rectangle([x, y, x + TH - 1, y + TH - 1], outline=(42, 51, 61))
    d.text((x + 6, y + 6), f"f{f}", fill=ACC, font=F_N)
    d.text((x, y + TH + 8), lab, fill=INK, font=F_S)

sheet.save(OUT)
print(f"wrote {OUT} {sheet.size}")
for f, lab in BEATS:
    print(f"  f{f:4d}  {lab}")
