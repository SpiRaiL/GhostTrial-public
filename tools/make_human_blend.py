"""Build a .blend holding every version of the human motion, side by side.

    /snap/bin/blender --background --python tools/make_human_blend.py -- <out.blend>

Every take in data/human_bvh_edited is imported, given renderable geometry, placed
in its own spot along X, colour-coded, and labelled on the floor beneath it. All of
them are left switched ON so the file opens showing the whole set — turn off the
ones you do not want in the outliner.

The camera sits on the +Y side, mirrored across the X axis from where it used to
be, because that is the angle the final clips use. The saved viewport is aligned to
it, so the file opens looking through the shot.

Bones get real geometry (a box per bone) rather than viewport-only bone display,
so the file renders as it looks.
"""

import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BVHDIR = os.environ.get("GT_BVHDIR", os.path.join(REPO, "data", "human_bvh_edited"))

SPACING = 2.2
PHYS_Y = -9.0        # the physics track sits on its own row, well clear of the rest
PHYS_SPACING = 3.2
# the balance line the indicator ball rides above and below
LINE_Z = 2.35
BALL_SCALE = 6.0     # metres of ball travel per metre of balance margin
# what each take is, for the floor label — keep this in step with the tools that
# produce them
WHAT = {
    "A_side_deepcrouch": "A\nraw capture",
    "A2_throw_flat": "A2\nthrow arm\nlevel",
    "A3_arm_clear": "A3\n+ arm clear\nof thigh",
    "A4_natural_speed": "A4\n+ natural\nuppercut speed",
    "A5_stable_stance": "A5\n+ shallower\ncrouch, slower",
    "A6_toein": "A6\n+ toes\nbrought in",
    "A7_narrow": "A7\n+ narrow\nstance",
    "A8_flatfeet": "A8\n+ narrow,\nflat feet",
    "B_side_hold": "B\nsecond side\ntake",
    "C_angle45_besthold": "C\n45 degree\ntake",
}
# distinct, and ordered so the progression reads left to right
COLOURS = [
    (0.62, 0.64, 0.68), (0.35, 0.62, 0.90), (0.30, 0.78, 0.62), (0.95, 0.72, 0.30),
    (0.90, 0.45, 0.40), (0.72, 0.50, 0.88), (0.40, 0.80, 0.85), (0.85, 0.55, 0.75),
    (0.55, 0.75, 0.40), (0.80, 0.80, 0.45),
]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 60

names = sorted(f for f in os.listdir(BVHDIR) if f.endswith(".bvh"))
span = (len(names) - 1) * SPACING

# ── floor ────────────────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=max(40.0, span + 14), location=(span / 2, 0, 0))
floor = bpy.context.object
floor.name = "floor"
fm = bpy.data.materials.new("floor_mat")
fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.17, .18, .21, 1)
fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
floor.data.materials.append(fm)

TORSO = {"Hips", "Spine1", "Spine2", "Chest"}


def build_body(arm, mat):
    made = []
    for bone in arm.data.bones:
        if bone.length < 1e-4 or bone.name == "Root":
            continue
        bpy.ops.mesh.primitive_cube_add(size=1)
        seg = bpy.context.object
        seg.name = f"{arm.name}_seg_{bone.name}"
        seg.data.materials.append(mat)
        t = 0.55 if bone.name in TORSO else 0.20
        seg.scale = (max(bone.length * t, 0.014), bone.length, max(bone.length * t, 0.014))
        bpy.ops.object.transform_apply(scale=True)
        seg.parent = arm
        seg.parent_type = "BONE"
        seg.parent_bone = bone.name
        seg.location = (0, -bone.length * 0.5, 0)
        made.append(seg)
    head = arm.data.bones.get("Head")
    if head:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=24, ring_count=16)
        sk = bpy.context.object
        sk.name = f"{arm.name}_seg_Head"
        sk.data.materials.append(mat)
        sk.scale = (0.085, 0.105, 0.098)
        bpy.ops.object.shade_smooth()
        sk.parent = arm
        sk.parent_type = "BONE"
        sk.parent_bone = "Head"
        sk.location = (0, -head.length * 0.35, 0.01)
        made.append(sk)
    return made


longest = None
for i, fn in enumerate(names):
    tag = fn[:-10] if fn.endswith("_human.bvh") else fn[:-4]
    before = set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(filepath=os.path.join(BVHDIR, fn), global_scale=0.01,
                            use_fps_scale=False, update_scene_fps=False,
                            update_scene_duration=True, rotate_mode="NATIVE")
    arm = (set(bpy.context.scene.objects) - before).pop()
    arm.name = tag
    arm.location.x = i * SPACING
    arm.data.display_type = "OCTAHEDRAL"

    rgb = COLOURS[i % len(COLOURS)]
    mat = bpy.data.materials.new(f"mat_{tag}")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = 0.45
    # solid-mode viewport colour: set BOTH the material's viewport colour and the
    # object colour, and switch the shading to Object below, so the takes are
    # distinguishable without turning rendered preview on
    mat.diffuse_color = (*rgb, 1.0)
    body = build_body(arm, mat)
    for ob in body + [arm]:
        ob.color = (*rgb, 1.0)

    # floor label, lying flat and facing the camera side (+Y)
    bpy.ops.object.text_add(location=(i * SPACING, 1.35, 0.004))
    txt = bpy.context.object
    txt.name = f"label_{tag}"
    txt.data.body = WHAT.get(tag, tag)
    txt.data.size = 0.20
    txt.data.align_x = "CENTER"
    txt.data.align_y = "TOP"
    txt.color = (*rgb, 1.0)
    txt.rotation_euler = (0, 0, 3.14159265)      # readable from +Y
    lm = bpy.data.materials.new(f"lab_{tag}")
    lm.use_nodes = True
    lb = lm.node_tree.nodes["Principled BSDF"]
    lb.inputs["Base Color"].default_value = (*rgb, 1.0)
    lb.inputs["Emission Color"].default_value = (*rgb, 1.0)
    lb.inputs["Emission Strength"].default_value = 4.0
    txt.data.materials.append(lm)

    coll = bpy.data.collections.new(f"take_{tag}")
    scene.collection.children.link(coll)
    for ob in body + [arm, txt]:
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        coll.objects.link(ob)
    # every take left ON — the whole set is meant to be visible on open
    n = int(arm.animation_data.action.frame_range[1])
    if longest is None or n > longest[1]:
        longest = (arm, n)
    print(f"imported {fn}: 1..{n}  at x={i * SPACING:.1f}  {WHAT.get(tag, tag)}")

# ── cameras, on the +Y side (mirrored over the X axis from before) ────────────
def add_cam(name, loc, target_loc, lens):
    bpy.ops.object.empty_add(location=target_loc)
    t = bpy.context.object
    t.name = f"{name}_target"
    bpy.ops.object.camera_add(location=loc)
    c = bpy.context.object
    c.name = name
    c.data.lens = lens
    k = c.constraints.new("TRACK_TO")
    k.target = t
    k.track_axis = "TRACK_NEGATIVE_Z"
    k.up_axis = "UP_Y"
    return c


# 24 mm and far enough back to fit the whole line-up: a 40 mm lens only covers
# about 12 m at this distance and the row is nearly 20 m wide
cam_all = add_cam("cam_all_takes", (span / 2, span * 0.95 + 6.0, 3.2),
                  (span / 2, 0, 0.95), 24)
cam_main = add_cam("cam_single", (longest[0].location.x - 1.2, 3.4, 1.45),
                   (longest[0].location.x, 0, 0.95), 42)
# orthographic inspection view — no perspective, so poses can be compared
# straight across, tilted down enough to read the floor labels
cam_ortho = add_cam("cam_ortho", (span / 2, 26.0, 6.5), (span / 2, 0.4, 0.55), 50)
cam_ortho.data.type = "ORTHO"
cam_ortho.data.ortho_scale = span + 5.0
scene.camera = cam_all

bpy.ops.object.light_add(type="AREA", location=(span / 2, 5.0, 6.0))
bpy.context.object.data.energy = 1400
bpy.context.object.data.size = 12
bpy.ops.object.light_add(type="AREA", location=(span / 2, -5.0, 4.0))
bpy.context.object.data.energy = 500
bpy.context.object.data.size = 12
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.05, .06, .07, 1)

scene.frame_start = 1
scene.frame_end = longest[1]

# open looking through the shot, in SOLID shading coloured per object
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for sp in area.spaces:
            if sp.type != "VIEW_3D":
                continue
            sp.shading.type = "SOLID"
            sp.shading.color_type = "OBJECT"
            sp.region_3d.view_perspective = "CAMERA"


# ── PHYSICS TRACK ────────────────────────────────────────────────────────────
# A separate row. These are G1 results, not human motion: body transforms come
# straight out of MuJoCo (tools/export_g1_bodies.py) so what is drawn is exactly
# what was measured. Each carries a ball riding above or below a stability line —
# above means the centre of mass is inside the feet and the pose could be held,
# below means it is falling. It is the balance-margin graph, in real time.
import numpy as np                                              # noqa: E402

PHYS = [
    ("P0_unbalanced", "PHYSICS TRACK\nP0  as captured\n(unbalanced)", (0.90, 0.42, 0.38)),
    ("P1_balanced", "PHYSICS TRACK\nP1  leaned to balance\n+ soles flattened", (0.35, 0.85, 0.55)),
    ("P2_A9_reference", "PHYSICS TRACK\nA9 REFERENCE\nsmoothed, what we ask for", (0.45, 0.70, 0.95)),
    ("P3_A9_policy", "PHYSICS TRACK\nA9 POLICY\nwhat it actually does", (0.95, 0.80, 0.35)),
]
PDIR = os.path.join(REPO, "data", "g1_bodies")
for pi, (tag, label, rgb) in enumerate(PHYS):
    f = os.path.join(PDIR, tag + ".npz")
    if not os.path.exists(f):
        print(f"skip {tag}: no {f}")
        continue
    z = np.load(f, allow_pickle=True)
    pos, mat, size, margin = z["pos"], z["mat"], z["size"], z["margin"]
    T_, NB = pos.shape[0], pos.shape[1]
    ox = pi * PHYS_SPACING

    mat_p = bpy.data.materials.new(f"mat_{tag}")
    mat_p.use_nodes = True
    bp = mat_p.node_tree.nodes["Principled BSDF"]
    bp.inputs["Base Color"].default_value = (*rgb, 1.0)
    bp.inputs["Roughness"].default_value = 0.4
    mat_p.diffuse_color = (*rgb, 1.0)

    coll = bpy.data.collections.new(f"phys_{tag}")
    scene.collection.children.link(coll)
    made = []
    for k in range(NB):
        bpy.ops.mesh.primitive_cube_add(size=1)
        cu = bpy.context.object
        cu.name = f"{tag}_body{k:02d}"
        cu.scale = tuple(float(v) * 2.0 for v in size[k])
        bpy.ops.object.transform_apply(scale=True)
        cu.data.materials.append(mat_p)
        cu.color = (*rgb, 1.0)
        cu.rotation_mode = "QUATERNION"
        for fr in range(T_):
            R = mat[fr, k]
            q = __import__("mathutils").Matrix(((R[0][0], R[0][1], R[0][2]),
                                                (R[1][0], R[1][1], R[1][2]),
                                                (R[2][0], R[2][1], R[2][2]))).to_quaternion()
            cu.location = (float(pos[fr, k, 0]) + ox, float(pos[fr, k, 1]) + PHYS_Y,
                           float(pos[fr, k, 2]))
            cu.rotation_quaternion = q
            cu.keyframe_insert("location", frame=fr + 1)
            cu.keyframe_insert("rotation_quaternion", frame=fr + 1)
        made.append(cu)

    # the stability line, and the ball that rides it
    bpy.ops.mesh.primitive_cube_add(size=1, location=(ox, PHYS_Y, LINE_Z))
    line = bpy.context.object
    line.name = f"{tag}_stability_line"
    line.scale = (1.5, 0.02, 0.006)
    lmat = bpy.data.materials.new(f"line_{tag}")
    lmat.use_nodes = True
    ln = lmat.node_tree.nodes["Principled BSDF"]
    ln.inputs["Base Color"].default_value = (0.85, 0.85, 0.9, 1)
    ln.inputs["Emission Color"].default_value = (0.85, 0.85, 0.9, 1)
    ln.inputs["Emission Strength"].default_value = 2.0
    line.data.materials.append(lmat)
    lmat.diffuse_color = (0.9, 0.9, 0.95, 1)
    line.color = (0.9, 0.9, 0.95, 1)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.13, segments=24, ring_count=14,
                                         location=(ox, PHYS_Y, LINE_Z))
    ball = bpy.context.object
    ball.name = f"{tag}_balance_ball"
    bpy.ops.object.shade_smooth()
    bmat = bpy.data.materials.new(f"ball_{tag}")
    bmat.use_nodes = True
    bn = bmat.node_tree.nodes["Principled BSDF"]
    ball.data.materials.append(bmat)
    for fr in range(T_):
        m = float(margin[fr])
        good = m > 0
        col = (0.30, 0.85, 0.45, 1) if good else (0.92, 0.35, 0.30, 1)
        ball.location = (ox, PHYS_Y, LINE_Z + max(min(m, 0.10), -0.12) * BALL_SCALE)
        ball.keyframe_insert("location", frame=fr + 1)
        bn.inputs["Base Color"].default_value = col
        bn.inputs["Emission Color"].default_value = col
        bn.inputs["Emission Strength"].default_value = 2.2
        bn.inputs["Base Color"].keyframe_insert("default_value", frame=fr + 1)
        bn.inputs["Emission Color"].keyframe_insert("default_value", frame=fr + 1)
        ball.color = col
        ball.keyframe_insert('color', frame=fr + 1)

    bpy.ops.object.text_add(location=(ox, PHYS_Y + 1.35, 0.004))
    pt = bpy.context.object
    pt.name = f"label_{tag}"
    pt.data.body = label
    pt.data.size = 0.20
    pt.data.align_x = "CENTER"
    pt.data.align_y = "TOP"
    pt.rotation_euler = (0, 0, 3.14159265)
    pt.data.materials.append(lmat)
    pt.color = (*rgb, 1.0)

    for ob in made + [line, ball, pt]:
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        coll.objects.link(ob)
    print(f"physics track: {tag}  {T_} frames, {NB} bodies at x={ox:.1f} y={PHYS_Y}")

# a camera that takes in both rows
# high and angled down, so the two rows separate into distinct bands instead of
# the far one hiding behind the near one
add_cam("cam_both_tracks", (span / 2, 20.0, 15.0), (span / 2, PHYS_Y / 2 - 1.0, 0.9), 28)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f"saved {OUT}  ({len(names)} takes, all visible, camera on +Y)")
