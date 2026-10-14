#!/usr/bin/env bash
# Download MediaPipe Tasks Vision (JS + wasm) and the pose/hand models into vendor/,
# so the experiment can run without internet. Then set mediapipe.source = 'local' in js/config.js.
#   bash tools/fetch_mediapipe_assets.sh            (from the web_experiment folder)
set -euo pipefail
cd "$(dirname "$0")/.."
V="${MP_VERSION:-0.10.14}"
CDN="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${V}"
OUT="vendor/mediapipe"
mkdir -p "$OUT/wasm" "$OUT/models"
curl -fsSL "$CDN/vision_bundle.mjs" -o "$OUT/vision_bundle.mjs"
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  curl -fsSL "$CDN/wasm/$f" -o "$OUT/wasm/$f"
done
M="https://storage.googleapis.com/mediapipe-models"
for v in lite full heavy; do
  curl -fsSL "$M/pose_landmarker/pose_landmarker_${v}/float16/1/pose_landmarker_${v}.task" -o "$OUT/models/pose_landmarker_${v}.task"
done
curl -fsSL "$M/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o "$OUT/models/hand_landmarker.task"
echo "Done. Files in $OUT:"; ls -lR "$OUT"
