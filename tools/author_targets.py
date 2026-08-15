#!/usr/bin/env python3
"""Author candidate targets that a whole-body controller can actually track.

    MUJOCO_GL=egl .venv/bin/python tools/author_targets.py --review

Why these knobs and not others. Comparing three clips whose trained reward we know:

    clip          knee/hip   sole gap   reward
    CTRL2 walk      1.48       1.8 cm     65.6
    A9              0.74       3.5 cm     38
    A11             0.45       2.8 cm     ~15

The ordering is exact. The motion SONIC tracks best flexes the KNEE more than the
hip; ours does the opposite — A11 bends at the hip 2.2x more than the knee and
sweeps 113.8 degrees of hip pitch against the walk's 48.7. Bowing at the waist to
get low, instead of sitting down into the knees, is the single strongest signal we
have for untrackability, and it is what David spotted by eye: "we bend at the hips
but not the knees to get the feet back on the floor".

So each candidate trades hip lean for knee flexion at constant pelvis height:

  --lean S     scale hip pitch toward its own median (S<1 = stand the torso up)
  --knee K     lower the pelvis K cm through the knees, ankle keeping the sole flat
  --flatten    roll/pitch the ankle to put the sole down where it is nearly down

Every edit is solved per frame by bisection against what MuJoCo measures, bounded
by the joint's real remaining travel, smoothed over time, and then re-seated so each
frame's soles sit exactly where the source clip's did. Those four rules are all
scar tissue: an unbounded solve threw the legs to 78 degrees, a per-frame solve
jittered the root hard enough to send inverse-dynamics torque to 14901% of limit,
and forgetting to re-seat floated three separate takes because leg edits move the
FEET, not the pelvis.
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_columns import joint_cols  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "vendor", "GR00T-WholeBodyControl", "gear_sonic", "data",
                   "assets", "robot_description", "mjcf", "g1_29dof_rev_1_0.xml")
MAX_K = np.deg2rad(30.0)

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
FEET = {s: [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
            == f"{s}_ankle_roll_link"] for s in ("left", "right")}
LEG = ("hip_pitch", "knee", "ankle_pitch")
GAIN = {"hip_pitch": 1.0, "knee": 2.0, "ankle_pitch": 1.0, "hip_roll": 1.0}


def corners(side):
    pts = []
    for g in FEET[side]:
        p, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    pts.append(p + R @ (np.array([a, b, c]) * sz))
    return np.array(pts)


def measure(root, quat, dof):
    """Everything worth judging a candidate on, per frame."""
    T = len(dof)
    out = dict(low=np.zeros((T, 2)), flat=np.zeros((T, 2)), com=np.zeros((T, 3)),
               pelvis=np.zeros(T), poly=[None] * T)
    for i in range(T):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], dof[i]
        mujoco.mj_forward(model, data)
        out["com"][i] = data.subtree_com[0]
        out["pelvis"][i] = data.qpos[2]
        cs = []
        for k, s in enumerate(("left", "right")):
            P = corners(s)
            out["low"][i, k] = P[:, 2].min()
            s4 = np.sort(P[:, 2])[:4]
            out["flat"][i, k] = s4.max() - s4.min()
            cs.append(P)
        out["poly"][i] = cs
    return out


def balance_margin(m):
    """Signed distance from CoM to the support polygon — the fig_static metric.

    Support is the union of the FULL soles of feet that are down, because a loaded
    foot settles flat; counting only the corners that happen to touch in a kinematic
    pose reported tilted feet as unsupported and produced a badly wrong answer once
    already.
    """
    from scipy.spatial import ConvexHull
    T = len(m["pelvis"])
    floor = np.percentile(m["low"].min(axis=1), 2)
    down = (m["low"] - floor) < 0.045
    marg = np.full(T, -np.inf)
    for i in range(T):
        pts = [m["poly"][i][k][:, :2] for k in range(2) if down[i, k]]
        if not pts:
            continue
        P = np.vstack(pts)
        try:
            hull = ConvexHull(P)
        except Exception:
            continue
        inside, best = True, np.inf
        for eq in hull.equations:
            dist = float(eq[:2] @ m["com"][i][:2] + eq[2])
            if dist > 0:
                inside = False
            best = min(best, abs(dist))
        marg[i] = best if inside else -best
    return marg, down


def headroom(dof, idx, i, sign, parts):
    room = MAX_K
    for side in ("left", "right"):
        for part in parts:
            n = f"{side}_{part}_joint"
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)
            q, step = dof[i][idx[n]], sign * GAIN[part]
            if step > 0:
                room = min(room, max(0.0, (hi - q) / step))
            elif step < 0:
                room = min(room, max(0.0, (q - lo) / -step))
    return room


def smooth_clamp(k, dof, idx, sign, fps, parts):
    w = max(3, int(round(0.25 * fps)) | 1)
    k = np.convolve(np.pad(k, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    return np.array([min(k[i], headroom(dof, idx, i, sign, parts)) for i in range(len(k))])


def narrow(dof, idx, root, quat, target_cm, fps):
    """Bring the feet together by hip roll, solved per frame against MuJoCo.

    The feet sit 52.6 cm apart laterally against 30.6 cm in the walk SONIC tracks.
    That straddle is what rolls the soles onto their edges and lifts the heels — not
    a hip-flexion limit, which has 80-131 degrees still unused. Standing the torso up
    does nothing for it.
    """
    T = len(dof)
    iL, iR = idx["left_hip_roll_joint"], idx["right_hip_roll_joint"]

    def width(i, k):
        q = dof[i].copy()
        q[iL] -= k
        q[iR] += k
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], q
        mujoco.mj_forward(model, data)
        yaw = Rotation.from_quat(quat[i][[1, 2, 3, 0]]).as_euler("zyx")[0]
        Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        L = Rz @ corners("left").mean(0)[:2]
        R = Rz @ corners("right").mean(0)[:2]
        return abs(L[1] - R[1])

    sign = 1.0 if width(0, np.deg2rad(1.0)) < width(0, 0.0) else -1.0
    ks = np.zeros(T)
    for i in range(T):
        if width(i, 0.0) <= target_cm / 100.0:
            continue
        lo, hi = 0.0, min(MAX_K, headroom(dof, idx, i, sign, ("hip_roll",)))
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if width(i, sign * mid) > target_cm / 100.0:
                lo = mid
            else:
                hi = mid
        ks[i] = hi
    w = max(3, int(round(0.25 * fps)) | 1)
    ks = np.convolve(np.pad(ks, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    for i in range(T):
        ks[i] = min(ks[i], headroom(dof, idx, i, sign, ("hip_roll",)))
        dof[i, iL] -= sign * ks[i]
        dof[i, iR] += sign * ks[i]
    return dof


def author(src, lean=1.0, knee_cm=0.0, flatten=False, narrow_cm=0.0, fps=60.0):
    df = pd.read_csv(src)
    jc = joint_cols(df)
    tc = ["root_translateX", "root_translateY", "root_translateZ"]
    rc = ["root_rotateX", "root_rotateY", "root_rotateZ"]
    idx = {c[:-4]: i for i, c in enumerate(jc)}
    root = df[tc].values / 100.0
    quat = Rotation.from_euler("xyz", df[rc].values, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    T = len(df)
    before = measure(root, quat, dof)
    orig_low = before["low"].min(axis=1)

    # 1. stand the torso up: pull hip pitch toward its own median
    if lean != 1.0:
        for side in ("left", "right"):
            col = idx[f"{side}_hip_pitch_joint"]
            med = np.median(dof[:, col])
            dof[:, col] = med + (dof[:, col] - med) * lean

    # 2. put the height back through the knees
    if knee_cm > 0:
        probe = dof.copy()
        for side in ("left", "right"):
            probe[0, idx[f"{side}_knee_joint"]] += np.deg2rad(2.0)
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[0], quat[0], probe[0]
        mujoco.mj_forward(model, data)
        lowered = data.qpos[2] - corners("left")[:, 2].min()
        data.qpos[7:] = dof[0]
        mujoco.mj_forward(model, data)
        base = data.qpos[2] - corners("left")[:, 2].min()
        sign = 1.0 if lowered < base else -1.0

        ks = np.zeros(T)
        for i in range(T):
            lo, hi = 0.0, min(MAX_K, headroom(dof, idx, i, sign, LEG))
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                q = dof[i].copy()
                for side in ("left", "right"):
                    for part in LEG:
                        q[idx[f"{side}_{part}_joint"]] += sign * mid * GAIN[part]
                data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], q
                mujoco.mj_forward(model, data)
                h = data.qpos[2] - min(corners("left")[:, 2].min(), corners("right")[:, 2].min())
                if h > before["pelvis"][i] - before["low"][i].min() - knee_cm / 100.0:
                    lo = mid
                else:
                    hi = mid
            ks[i] = hi
        ks = smooth_clamp(ks, dof, idx, sign, fps, LEG)
        for i in range(T):
            for side in ("left", "right"):
                for part in LEG:
                    dof[i, idx[f"{side}_{part}_joint"]] += sign * ks[i] * GAIN[part]

    # 3. bring the feet together
    if narrow_cm > 0:
        dof = narrow(dof, idx, root, quat, narrow_cm, fps)

    # 4. roll the ankle so a nearly-down sole lies flat
    if flatten:
        for i in range(T):
            for side in ("left", "right"):
                col = idx[f"{side}_ankle_roll_joint"]
                # clamp the search to the joint's real range — the first version
                # swept a fixed +-0.12 rad and drove ankle_roll 6.9 deg past its stop
                # on hundreds of frames, trading one infeasibility for another
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}_ankle_roll_joint")
                jlo, jhi = (model.jnt_range[jid] if model.jnt_limited[jid]
                            else (-np.inf, np.inf))
                best, bq = None, dof[i, col]
                for delta in np.linspace(-0.12, 0.12, 25):
                    q = dof[i].copy()
                    q[col] = float(np.clip(bq + delta, jlo, jhi))
                    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[i], quat[i], q
                    mujoco.mj_forward(model, data)
                    P = corners(side)
                    s4 = np.sort(P[:, 2])[:4]
                    gap = s4.max() - s4.min()
                    if best is None or gap < best[0]:
                        best = (gap, q[col])
                dof[i, col] = best[1]

    # 4. re-seat: leg edits move the feet, so put each frame's soles back
    after = measure(root, quat, dof)
    root = root.copy()
    root[:, 2] -= after["low"].min(axis=1) - orig_low

    out = df.copy()
    out[jc] = np.rad2deg(dof)
    out[tc[2]] = root[:, 2] * 100.0
    return out, jc


def report(path, label):
    df = pd.read_csv(path)
    jc = joint_cols(df)
    root = df[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    quat = Rotation.from_euler(
        "xyz", df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values,
        degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof = np.deg2rad(df[jc].values)
    m = measure(root, quat, dof)
    marg, down = balance_margin(m)
    hr = np.rad2deg(dof[:, [i for i, c in enumerate(jc) if "hip_pitch" in c]])
    kr = np.rad2deg(dof[:, [i for i, c in enumerate(jc) if "knee" in c]])
    hrng, krng = hr.max() - hr.min(), kr.max() - kr.min()
    fin = marg[np.isfinite(marg)]
    return dict(label=label, path=path,
                knee_hip=krng / max(hrng, 1e-9), hip_range=hrng, knee_range=krng,
                flat=float(np.median(m["flat"])) * 100,
                both_down=100 * down.all(axis=1).mean(),
                balanced=100 * float((marg > 0).mean()),
                margin_med=float(np.median(fin)) * 100 if len(fin) else float("nan"),
                pelvis=float(m["pelvis"].mean()) * 100, frames=len(df), margin=marg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gemx_g1_retimed/A9_smooth.csv")
    ap.add_argument("--outdir", default="data/gemx_g1_retimed")
    args = ap.parse_args()

    # A9 is the base, not A11: it scored reward 38 against A11's ~15, so the faster,
    # deeper edit made things worse and is not worth building on.
    # Adding a CONSTANT knee bend was tried and does not help: it lowers the pelvis
    # but leaves knee RANGE untouched, so knee/hip got worse (0.24 and 0.35 against
    # A9's own 0.74). The lever that moves the ratio is hip lean — stand the torso up
    # and the same motion is carried by the knees.
    CANDS = [
        ("T1_upright", dict(lean=0.65, flatten=True)),
        ("T2_tall",    dict(lean=0.45, flatten=True)),
        ("T3_tall_sit", dict(lean=0.45, knee_cm=3.0, flatten=True)),
        ("T4_narrow", dict(lean=0.65, narrow_cm=34.0, flatten=True)),
    ]
    rows = [report(args.src, "A9 (source, reward 38)")]
    for name, kw in CANDS:
        out, _ = author(args.src, **kw)
        p = os.path.join(args.outdir, f"{name}.csv")
        out.to_csv(p, index=False, float_format="%.6f")
        rows.append(report(p, name))
    rows.append(report("data/csv_frozen/ctrl2/CTRL2_walk.csv", "CTRL2 walk (reward 65.6)"))

    print(f"\n{'candidate':26s}{'knee/hip':>9s}{'hip°':>7s}{'knee°':>7s}"
          f"{'sole gap':>10s}{'both down':>11s}{'balanced':>10s}{'margin':>9s}")
    for r in rows:
        print(f"{r['label']:26s}{r['knee_hip']:9.2f}{r['hip_range']:7.0f}{r['knee_range']:7.0f}"
              f"{r['flat']:9.1f}cm{r['both_down']:10.0f}%{r['balanced']:9.0f}%"
              f"{r['margin_med']:+8.1f}cm")
    np.save("/tmp/target_margins.npy",
            np.array([r["margin"] for r in rows], dtype=object), allow_pickle=True)
    with open("/tmp/target_labels.txt", "w") as f:
        f.write("\n".join(r["label"] for r in rows))
