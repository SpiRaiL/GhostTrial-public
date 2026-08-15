"""Dump the Mixamo rest-pose skeleton so poses can be authored against it.

    /snap/bin/blender --background --python tools/inspect_rig.py -- <rig.fbx>
"""

import sys
import bpy

src = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, automatic_bone_orientation=True)
arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")

print("\n=== rest pose (world space, metres after 0.01 armature scale) ===")
print(f"{'bone':32} {'head (x,y,z)':>30} {'len':>7}  parent")
mw = arm.matrix_world
for b in arm.data.bones:
    h = mw @ b.head_local
    t = mw @ b.tail_local
    ln = (t - h).length
    par = b.parent.name if b.parent else "-"
    print(f"{b.name:32} {h.x:9.3f}{h.y:9.3f}{h.z:9.3f} {ln:9.3f}  {par}")
