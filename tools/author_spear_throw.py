"""Hand-author the spear throw on the Mixamo rig and export BVH.

    /snap/bin/blender --background --python tools/author_spear_throw.py -- <out.bvh>

Why author it rather than lift it from video: we control the pose exactly, so we
can key the move *inside* the G1's comfortable joint range instead of discovering
afterwards that the retarget pinned both ankles (see the progress report §6).

Rig conventions, measured from the Mixamo rest pose (tools/inspect_rig.py):
    armature space is CENTIMETRES (object scale 0.01), hips rest at ~104 cm
    +X = character's left,  +Z = up,  -Y = forward (the way they face)
    every bone's local +Y runs head->tail, so local-Y rotation is TWIST

Posing is a hybrid, because pure aim-at-a-direction cannot express twist:
    hips          explicit world matrix  (yaw / pitch / roll + translation)
    spine chain   explicit local euler   (bend, twist, side-bend)
    limbs         aim a bone down a world direction, with a roll reference
"""

import math
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
RIG = argv[1] if len(argv) > 1 else "raw/mixamo/uppercut.fbx"

# ── direction shorthands (armature space) ──────────────────────────────────────
FWD, BACK = Vector((0, -1, 0)), Vector((0, 1, 0))
UP, DOWN = Vector((0, 0, 1)), Vector((0, 0, -1))
LEFT, RIGHT = Vector((1, 0, 0)), Vector((-1, 0, 0))


def D(*parts):
    """Blend direction shorthands: D((FWD,2),(DOWN,1)) -> mostly forward, some down."""
    v = Vector((0, 0, 0))
    for d, w in parts:
        v += d * w
    return v.normalized()


# ── rig loading ────────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=RIG, automatic_bone_orientation=True)
arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")

# drop the donor clip; we are authoring from the rest pose
arm.animation_data_clear()
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
    pb.matrix_basis = Matrix()

P = "mixamorig:"
scene = bpy.context.scene
scene.render.fps = 30


def upd():
    bpy.context.view_layer.update()


def W2L(v):
    """World direction -> armature-local.

    The FBX importer leaves the Y-up->Z-up conversion on the armature OBJECT, so
    armature-local space is Y-up / +Z-forward / centimetres while the world is
    Z-up / -Y-forward / metres. `pb.matrix` lives in armature-local space, so
    every direction written in readable world terms has to come through here.
    Measured: world +Z -> local +Y, world -Y -> local +Z, world +X -> local +X.
    """
    return (arm.matrix_world.to_3x3().inverted() @ Vector(v)).normalized()


def aim(name, direction, roll=UP):
    """Point a bone down `direction` (WORLD space), using `roll` to fix twist."""
    pb = arm.pose.bones[P + name]
    d = W2L(direction)
    r = W2L(roll)
    x = r.cross(d)
    if x.length < 1e-5:
        x = Vector((1, 0, 0)).cross(d)
        if x.length < 1e-5:
            x = Vector((0, 1, 0)).cross(d)
    x.normalize()
    # right-handed basis: columns are the bone's local X, Y(=d), Z, and X x Y = Z.
    # Getting this cross product backwards yields a reflection (det -1), which
    # silently mirrors every aimed bone — it cost an hour, so: do not "simplify".
    z = x.cross(d).normalized()
    m = Matrix((x, d, z)).transposed().to_4x4()
    m.translation = pb.matrix.translation
    pb.matrix = m
    upd()


def spine(name, bend=0.0, twist=0.0, side=0.0):
    """Local rotation for a spine-chain bone, in degrees. +bend = lean forward."""
    pb = arm.pose.bones[P + name]
    pb.rotation_mode = "YXZ"
    pb.rotation_euler = (math.radians(bend), math.radians(twist), math.radians(side))
    upd()
    pb.rotation_mode = "QUATERNION"
    upd()


def hips(pos, yaw=0.0, pitch=0.0, roll=0.0):
    """Root placement. pos is (x, y, z) in WORLD METRES, angles in degrees.

    yaw is about world up, pitch about world X (lean forward/back).
    """
    pb = arm.pose.bones[P + "Hips"]
    Mw = arm.matrix_world.to_3x3()
    Rw = (Matrix.Rotation(math.radians(yaw), 3, "Z")
          @ Matrix.Rotation(math.radians(pitch), 3, "X")
          @ Matrix.Rotation(math.radians(roll), 3, "Y"))
    Rl = Mw.inverted() @ Rw @ Mw                      # same rotation, local coords
    m = (Rl @ arm.data.bones[P + "Hips"].matrix_local.to_3x3()).to_4x4()
    m.translation = arm.matrix_world.inverted() @ Vector(pos)
    pb.matrix = m
    upd()


KEYED = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
         "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
         "RightShoulder", "RightArm", "RightForeArm", "RightHand",
         "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
         "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]


def key(frame):
    for n in KEYED:
        pb = arm.pose.bones[P + n]
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        if n == "Hips":
            pb.keyframe_insert("location", frame=frame)


# ── the poses ──────────────────────────────────────────────────────────────────
# Right-handed throw. The character faces -Y; left foot leads. Beat timings are
# scaled from the reference (raw/mk-bts-getoverhere/README.md): a long coil, a
# very fast snap, then a held extension.

def stance(reach=0.0, coil=0.0):
    """Bladed fighting stance. `coil` winds the torso back, `reach` throws it forward."""
    yaw = -28 + coil * 22 - reach * 46          # torso opens away, then drives through
    drop = 8 + coil * 2 + reach * 4
    lean = coil * -4 + reach * 9
    shift = -reach * 14 + coil * 4               # weight back on the coil, forward on the throw

    hips((0, shift / 100.0, (104 - drop) / 100.0), yaw=yaw * 0.45, pitch=lean * 0.3)
    spine("Spine", bend=lean * 0.3, twist=yaw * 0.20)
    spine("Spine1", bend=lean * 0.3, twist=yaw * 0.20)
    spine("Spine2", bend=lean * 0.4, twist=yaw * 0.15)
    spine("Neck", bend=-lean * 0.4, twist=-yaw * 0.25)
    spine("Head", bend=-lean * 0.3, twist=-yaw * 0.30)

    # legs — left foot forward, both knees bent, feet flat and planted
    aim("LeftUpLeg", D((DOWN, 3.0), (FWD, 0.95), (LEFT, 0.62)), roll=FWD)
    aim("LeftLeg", D((DOWN, 3.0), (BACK, 0.35), (LEFT, 0.10)), roll=FWD)
    aim("LeftFoot", D((FWD, 1.0), (DOWN, 0.30)), roll=UP)
    aim("LeftToeBase", FWD, roll=UP)
    aim("RightUpLeg", D((DOWN, 3.0), (BACK, 0.95), (RIGHT, 0.62)), roll=FWD)
    aim("RightLeg", D((DOWN, 3.0), (FWD, 0.30), (RIGHT, 0.10)), roll=FWD)
    aim("RightFoot", D((FWD, 1.0), (DOWN, 0.45)), roll=UP)
    aim("RightToeBase", FWD, roll=UP)

    # left arm — guard, pulled in tighter as the right arm extends
    aim("LeftShoulder", D((LEFT, 1.0), (UP, 0.15)), roll=UP)
    aim("LeftArm", D((DOWN, 1.0), (FWD, 0.55 - reach * 0.35), (LEFT, 0.95)), roll=FWD)
    aim("LeftForeArm", D((FWD, 1.0), (UP, 0.8), (RIGHT, 0.30 + reach * 0.5)), roll=UP)
    aim("LeftHand", D((FWD, 1.0), (UP, 0.35)), roll=UP)

    # right arm — the throw
    aim("RightShoulder", D((RIGHT, 1.0), (UP, 0.15)), roll=UP)
    aim("RightArm",
        D((BACK, 0.30 + coil * 1.1 - reach * 0.30),
          (DOWN, 1.0 - coil * 0.55 - reach * 0.98),
          (RIGHT, 0.60 + coil * 0.30 - reach * 0.18),
          (UP, reach * 0.10),
          (FWD, reach * 1.9)),
        roll=UP)
    aim("RightForeArm",
        D((FWD, 0.55 + reach * 2.2 - coil * 0.55),
          (UP, 0.75 + coil * 0.9 - reach * 0.75),
          (RIGHT, 0.15 - reach * 0.15),
          (BACK, coil * 0.5)),
        roll=UP)
    aim("RightHand", D((FWD, 1.0), (UP, 0.30 - reach * 0.30)), roll=UP)


# frame : (reach, coil) — reach 0..1 extends the throw, coil 0..1 winds it back
BEATS = [
    (0,  0.00, 0.00),   # settled
    (10, 0.00, 0.18),   # weight settles back
    (32, 0.00, 1.00),   # fully coiled — arm drawn back and up
    (42, 1.00, 0.00),   # SNAP — 10 frames, 0.33 s
    (50, 0.98, 0.00),   # held at extension
    (64, 0.96, 0.00),   # still held
    (80, 0.00, 0.05),   # recovered to stance
]

for f, reach, coil in BEATS:
    stance(reach=reach, coil=coil)
    key(f)
    if f == 0:
        mw = arm.matrix_world
        print("\n--- frame 0 pose, WORLD space (sanity check) ---")
        for n in ["Hips", "Head", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
                  "RightArm", "RightForeArm", "RightHand"]:
            pb = arm.pose.bones[P + n]
            h, t = mw @ pb.head, mw @ pb.tail
            d = (t - h).normalized()
            print(f"  {n:13} head z {h.z:6.3f}  dir ({d.x:5.2f},{d.y:5.2f},{d.z:5.2f})")
        print()

scene.frame_start, scene.frame_end = 0, BEATS[-1][0]

# Blender's default bezier handles overshoot between distant keys; on the snap
# that produced a 12.3 m/s hand against a 5.8 m/s human reference. Auto-clamped
# handles cannot exceed the keyed values.
action = arm.animation_data.action
fcurves = []
if hasattr(action, "fcurves"):
    fcurves = list(action.fcurves)
else:
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                fcurves.extend(bag.fcurves)
for fc in fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "BEZIER"
        kp.handle_left_type = "AUTO_CLAMPED"
        kp.handle_right_type = "AUTO_CLAMPED"
    fc.update()

# strip the mixamorig: prefix so GMR's nokov BVH config matches
for b in arm.data.bones:
    b.name = b.name.replace(P, "")

bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_anim.bvh(filepath=OUT, frame_start=scene.frame_start,
                        frame_end=scene.frame_end, root_transform_only=True,
                        global_scale=1.0)
print(f"wrote {OUT}  frames {scene.frame_start}..{scene.frame_end}")
