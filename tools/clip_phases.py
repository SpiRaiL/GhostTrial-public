"""Dump world-space hand/hip trajectories from an FBX so a clip can be segmented.

    /snap/bin/blender --background --python tools/clip_phases.py -- <in.fbx>

Prints a per-frame table plus a guess at the phase boundaries, based on where
the hands are relative to the body and how fast they are moving. Used to pick
trim points for the strike (and, later, the seam for the uppercut splice).
"""

import sys
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src = argv[0]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, automatic_bone_orientation=True)

arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
scene = bpy.context.scene
start, end = (int(x) for x in arm.animation_data.action.frame_range)

TRACK = {
    "lhand": "mixamorig:LeftHand",
    "rhand": "mixamorig:RightHand",
    "hips": "mixamorig:Hips",
    "head": "mixamorig:Head",
}

rows = []
for f in range(start, end + 1):
    scene.frame_set(f)
    mw = arm.matrix_world
    p = {k: mw @ arm.pose.bones[b].head for k, b in TRACK.items()}
    rows.append((f, p))

print(f"\n=== {src}  frames {start}..{end} @ {scene.render.fps} fps\n")
print(f"{'frame':>5} {'t(s)':>6} {'hand_fwd':>9} {'hand_up':>8} {'hand_sep':>9} "
      f"{'hip_up':>7} {'hand_speed':>11}")

fps = scene.render.fps
prev = None
series = []
for f, p in rows:
    mid = (p["lhand"] + p["rhand"]) / 2
    hips = p["hips"]
    # Mixamo: -Y is forward after import, Z is up
    fwd = -(mid.y - hips.y)
    up = mid.z - hips.z
    sep = (p["lhand"] - p["rhand"]).length
    speed = 0.0 if prev is None else (mid - prev).length * fps
    prev = mid
    series.append((f, fwd, up, sep, hips.z, speed))
    print(f"{f:5d} {(f-start)/fps:6.2f} {fwd:9.3f} {up:8.3f} {sep:9.3f} "
          f"{hips.z:7.3f} {speed:11.3f}")

peak = max(series, key=lambda r: r[5])
reach = max(series, key=lambda r: r[1])
tuck = min(series[: reach[0] - start + 1], key=lambda r: r[1]) if reach[0] > start else series[0]

print(f"\nmost tucked (charge)  : frame {tuck[0]:3d}  t={(tuck[0]-start)/fps:.2f}s  fwd={tuck[1]:.3f}")
print(f"peak hand speed       : frame {peak[0]:3d}  t={(peak[0]-start)/fps:.2f}s  {peak[5]:.2f} m/s")
print(f"max forward reach     : frame {reach[0]:3d}  t={(reach[0]-start)/fps:.2f}s  fwd={reach[1]:.3f}")
print(f"\nsuggested strike window: {tuck[0]}..{reach[0]}  "
      f"({(reach[0]-tuck[0])/fps:.2f}s, {reach[0]-tuck[0]+1} frames)")
