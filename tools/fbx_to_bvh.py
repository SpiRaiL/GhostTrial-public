"""Convert a Mixamo FBX (skeleton only) to BVH, and report what's in it.

Run headless:
    /snap/bin/blender --background --python tools/fbx_to_bvh.py -- \
        <in.fbx> <out.bvh> [start] [end] [scale]

Mixamo bones carry a `mixamorig:` prefix, which is stripped here so the names
match GMR's BVH configs (`Hips`, `LeftUpLeg`, `LeftForeArm`, `LeftToeBase`, ...).

`scale` defaults to **1.0 = centimetres**, because GMR's BVH loader
(`utils/lafan1.py`) divides positions by 100 to get metres. Pass 0.01 for a
metres BVH if something downstream wants that instead.
"""

import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
trim_start = int(argv[2]) if len(argv) > 2 else None
trim_end = int(argv[3]) if len(argv) > 3 else None
scale = float(argv[4]) if len(argv) > 4 else 1.0

# start from an empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, automatic_bone_orientation=True)

arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
action = arm.animation_data.action
start, end = (int(x) for x in action.frame_range)
fps = bpy.context.scene.render.fps

print(f"\n=== {src}")
print(f"armature   : {arm.name}")
print(f"bones      : {len(arm.data.bones)}")
print(f"frames     : {start}..{end}  ({end - start + 1} frames)")
print(f"fps        : {fps}  -> {(end - start + 1) / fps:.2f} s")
print(f"scale      : {tuple(round(s, 4) for s in arm.scale)}")

root = next(b for b in arm.data.bones if b.parent is None)
print(f"root bone  : {root.name}")

def iter_fcurves(action):
    """Blender 4.4+ moved fcurves onto action layers/strips/channelbags."""
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


# root translation range, to see whether the clip travels
fcurves = [fc for fc in iter_fcurves(action) if fc.data_path.endswith("location")]
for fc in fcurves:
    vals = [fc.evaluate(f) for f in range(start, end + 1)]
    axis = "XYZ"[fc.array_index]
    print(f"root loc {axis} : min={min(vals):8.3f}  max={max(vals):8.3f}  travel={max(vals)-min(vals):7.3f}")

# strip the mixamorig: prefix so bone names match what retargeting configs expect
for bone in arm.data.bones:
    bone.name = bone.name.replace("mixamorig:", "")

bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
bpy.context.view_layer.objects.active = arm

out_start = trim_start if trim_start is not None else start
out_end = trim_end if trim_end is not None else end
print(f"exporting  : frames {out_start}..{out_end} ({out_end - out_start + 1}), scale {scale}")

bpy.ops.export_anim.bvh(
    filepath=dst,
    frame_start=out_start,
    frame_end=out_end,
    root_transform_only=True,
    global_scale=scale,
)
print(f"wrote {dst}\n")
