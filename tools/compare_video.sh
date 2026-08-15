#!/usr/bin/env bash
# Target vs policy, side by side, in one command — the review artefact for each iteration.
#
#   tools/compare_video.sh <name> <target.pkl> <rollout.csv> [outdir]
#
# e.g. tools/compare_video.sh b1 data/motion_lib_capture/robot/b1/B1_idle.pkl \
#          data/rollouts/b1_policy.csv reports
#
# Both sides go through tools/render_motion.py so the comparison is like for like,
# and both read their joint columns by NAME (tools/g1_columns.py) — rollout CSVs
# are written in IsaacLab order and rendering one as MuJoCo order draws a robot the
# policy never produced. That bug is why every earlier policy video looked broken.
#
# Ends with -movflags +faststart, without which the moov atom lands at the end of
# the file and a browser downloads the whole thing before showing a frame.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?short name, e.g. b1}"
PKL="${2:?target .pkl}"
CSV="${3:?rollout .csv}"
OUT="${4:-reports}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Each side is rendered at ITS OWN rate. The target is a motion_lib clip at 30 fps;
# the rollout is one row per control step at 50 Hz. Rendering both at a common rate
# desynchronises them within a second and puts the two robots at different points of
# the move — which is exactly what the first side-by-sides showed.
TGT_FPS=$(.venv/bin/python -c "import joblib,sys;d=joblib.load('$PKL');print(int(d[list(d)[0]]['fps']))")
POL_FPS=${POL_FPS:-50}
.venv/bin/python tools/pkl_to_csv.py "$PKL" "$TMP/target.csv"
MUJOCO_GL=egl .venv/bin/python tools/render_motion.py "$TMP/target.csv" "$TMP/t.mp4" --fps "$TGT_FPS" >/dev/null
MUJOCO_GL=egl .venv/bin/python tools/render_motion.py "$CSV"            "$TMP/p.mp4" --fps "$POL_FPS" >/dev/null
echo "  target $TGT_FPS fps, policy $POL_FPS fps"

ffmpeg -y -loglevel error -i "$TMP/t.mp4" -i "$TMP/p.mp4" -filter_complex \
  "[0:v]drawtext=text='TARGET':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[a];
   [1:v]drawtext=text='POLICY under physics':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[b];
   [a][b]hstack=inputs=2" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT/g1_${NAME}_vs_target.mp4"

# a still from the middle, for the report — seek by time, since select=eq(n,N)
# needs the frame index escaped through two levels of quoting and silently fails
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 \
      "$OUT/g1_${NAME}_vs_target.mp4")
ffmpeg -y -loglevel error -ss "$(echo "$DUR" | awk '{printf "%.2f", $1/2}')" \
  -i "$OUT/g1_${NAME}_vs_target.mp4" -vframes 1 "$OUT/g1_${NAME}_vs_target.png"

echo "wrote $OUT/g1_${NAME}_vs_target.mp4 and .png"
