#!/usr/bin/env python3
"""Render a G1 trajectory with Isaac Sim's RTX renderer.

Run inside the isaac-sim:6.0.0 container (NOT the training image):

    /isaac-sim/python.sh /gt/tools/render_isaac.py \
        --csv /gt/data/rollouts/a5_policy.csv --out /gt/media/frames/a5 --fps 50

Why a second image. Isaac Sim 5.1, which ships inside isaac-lab:2.3.2 where the
policy trains, cannot initialise its RTX renderer on this machine's Blackwell GPU
— it segfaults in librtx.scenedb.plugin.so at plugin startup. Isaac Sim 6.0.0
starts the same renderer on the same GPU without complaint. So physics and policy
run in 2.3.2, rendering happens here, and the trajectory CSV is the bridge.

This is kinematic playback of an already-simulated result: the poses came out of a
full-physics rollout, so nothing is re-simulated. Joint angles are written each
frame and only the renderer is stepped.

Needs --runtime=nvidia with NVIDIA_DRIVER_CAPABILITIES=all. `--gpus all` grants
only compute,utility — no graphics — and the renderer has no driver to find.
"""

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True, help="directory for the PNG frames")
ap.add_argument("--usd", default="/gt/vendor/GR00T-WholeBodyControl/gear_sonic/"
                                 "data/robots/g1/g1_29dof_textured.usd")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--every", type=int, default=1, help="render every Nth frame")
ap.add_argument("--cam", default="3.4,-3.0,0.35", help="camera offset from the robot")
ap.add_argument("--focal", type=float, default=24.0,
                help="focal length in mm. USD cameras default to 50mm, which is "
                     "about 13 deg vertical FOV — far too tight to fit a standing "
                     "robot without backing off ~7 m")
ap.add_argument("--look", type=float, default=-0.12,
                help="look-at height relative to the pelvis; negative keeps the "
                     "feet and the floor in frame, which is the whole point here")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

# plain numpy, not pandas — the isaac-sim image does not ship pandas
with open(args.csv) as fh:
    header = fh.readline().strip().split(",")
raw = np.loadtxt(args.csv, delimiter=",", skiprows=1)
ci = {n: i for i, n in enumerate(header)}
jcols = [c for c in header if c.endswith("_dof")]
root = raw[:, [ci["root_translateX"], ci["root_translateY"], ci["root_translateZ"]]] / 100.0
quat_xyzw = Rotation.from_euler(
    "xyz", raw[:, [ci["root_rotateX"], ci["root_rotateY"], ci["root_rotateZ"]]],
    degrees=True).as_quat()
quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
dof_deg = raw[:, [ci[c] for c in jcols]]

world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=args.usd, prim_path="/World/G1")
world.scene.add_default_ground_plane()

off = np.array([float(x) for x in args.cam.split(",")])
world.reset()

art = Articulation("/World/G1")
art.initialize()

# Capture through Replicator rather than isaacsim.sensors.camera.Camera — the
# latter's get_rgba() returned None on every frame here even after warm-up.
import omni.replicator.core as rep  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

stage = world.stage
cam_prim = UsdGeom.Camera.Define(stage, "/World/cam")
cam_prim.CreateFocalLengthAttr(args.focal)
cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())
cam_xform.ClearXformOpOrder()
op_t = cam_xform.AddTranslateOp()
op_r = cam_xform.AddRotateXYZOp()
# add_default_ground_plane() brings no lighting with it — without these the RTX
# render comes back essentially black
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(250.0)
dome.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(6000.0)
sun.CreateAngleAttr(1.5)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

# A visible floor, built as explicit USD geometry. The default ground plane renders
# white against a white dome so the robot appears to float, and rep.create.plane
# did not land where it was wanted — foot contact is the whole point of this video,
# so the floor has to read.
S = 60.0
quad = UsdGeom.Mesh.Define(stage, "/World/FloorVis")
quad.CreatePointsAttr([Gf.Vec3f(-S, -S, 0.003), Gf.Vec3f(S, -S, 0.003),
                       Gf.Vec3f(S, S, 0.003), Gf.Vec3f(-S, S, 0.003)])
quad.CreateFaceVertexCountsAttr([4])
quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
quad.CreateExtentAttr([Gf.Vec3f(-S, -S, 0.0), Gf.Vec3f(S, S, 0.01)])
quad.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.31, 0.34)])

render_product = rep.create.render_product("/World/cam", (args.width, args.height))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach([render_product])

# the CSV and the USD order their joints differently — map by name, never by index
usd_names = list(art.dof_names)
csv_names = [c[:-4] for c in jcols]          # strip the "_dof" suffix
order = [csv_names.index(n) for n in usd_names]
print(f"[isaac] {len(raw)} frames, {len(usd_names)} joints mapped by name", flush=True)

# gravity off: this is replay of a recorded result, not a new simulation
world.get_physics_context().set_gravity(0.0)

# The camera annotator produces nothing until the renderer has run a few times.
# Without this warm-up every get_rgba() comes back empty and the loop silently
# writes zero frames.
for _ in range(12):
    world.render()
rep.orchestrator.step(delta_time=0.0)
probe = rgb_annot.get_data()
print(f"[isaac] camera warm-up -> {np.asarray(probe).shape}", flush=True)

os.makedirs(args.out, exist_ok=True)
written = 0
for f in range(0, len(raw), args.every):
    art.set_world_poses(positions=np.array([root[f]]),
                        orientations=np.array([quat_wxyz[f]]))
    q = np.deg2rad(dof_deg[f][order])[None, :]
    art.set_joint_positions(q)
    art.set_joint_velocities(np.zeros((1, len(usd_names))))
    # Drive the targets to the same pose. rep.orchestrator.step() advances physics,
    # and the joint drives default to a target of zero — so without this they pull
    # against every pose we write and the robot progressively tears itself apart.
    # It only showed up at small --every, where there are enough steps to accumulate.
    art.set_joint_position_targets(q)

    eye = root[f] + off
    look = root[f] + np.array([0.0, 0.0, args.look])
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    # USD cameras look down -Z with +Y up; convert the look direction into the
    # XYZ euler that convention expects
    yaw = np.degrees(np.arctan2(fwd[1], fwd[0])) - 90.0
    pitch = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    op_t.Set(Gf.Vec3d(*eye.tolist()))
    op_r.Set(Gf.Vec3f(90.0 + pitch, 0.0, yaw))

    # delta_time=0 + pause_timeline: capture without advancing physics. A plain
    # orchestrator.step() re-simulates, and the robot falls away from the pose just
    # written (readback: asked z=0.659, got 0.128 by mid-clip). A plain
    # world.render() does not re-simulate but produces no annotator data at all.
    rep.orchestrator.step(delta_time=0.0)
    if f in (400, 470, 480, 490):
        got_p, got_q = art.get_world_poses()
        ge = Rotation.from_quat(np.asarray(got_q)[0][[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        print(f"[isaac] f{f}: askedZ={root[f][2]:.3f} gotZ={got_p[0][2]:.3f} "
              f"askedEul={np.round(Rotation.from_quat(quat_wxyz[f][[1,2,3,0]]).as_euler('xyz',degrees=True),1)} "
              f"gotEul={np.round(ge,1)}", flush=True)
    rgba = np.asarray(rgb_annot.get_data())
    if rgba is None or rgba.size == 0:
        if written == 0 and f > 20:
            raise SystemExit("[isaac] camera never produced a frame — aborting "
                             "rather than writing an empty video")
        continue
    Image.fromarray((rgba[:, :, :3]).astype(np.uint8)).save(
        os.path.join(args.out, f"f_{written:05d}.png"))
    written += 1
    if written % 100 == 0:
        print(f"[isaac] {written} frames", flush=True)

print(f"[isaac] wrote {written} frames to {args.out}", flush=True)
app.close()
