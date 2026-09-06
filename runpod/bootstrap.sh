#!/usr/bin/env bash
#
# One-shot RunPod job: retrain at 2048 on rasterisation-corrected targets, submit, stop.
#
# Every second this pod is alive costs money, so the script is written to spend
# as little of its life as possible not training:
#   - starts from exp_002's checkpoint, so no COCO weights and no cold start
#   - fetches only the two files it needs, in parallel with the pip install
#   - submits to Kaggle from inside the pod, so nothing has to be retrieved
#     afterwards and the pod can die immediately
#   - stops on convergence, not a clock: every clock-stopped run on this project
#     was still improving when its clock ran out, and Ultralytics checkpoints every
#     epoch so a torn-down pod keeps its work
#
# Requires KAGGLE_USERNAME and KAGGLE_KEY in the environment.
# Usage:  bash bootstrap.sh [PATIENCE]         (default 40 epochs)
#
set -euo pipefail

PATIENCE="${1:-40}"
RATE_PER_HOUR="${RATE_PER_HOUR:-0.34}"
REPO_URL="https://github.com/Cosmic-Witness/Sol"
BRANCH="claude/kaggle-credentials-setup-f7nudy"
WORK=/workspace
COMP=filament-segmentation-2026

start=$(date +%s)
say() { echo "[$(printf '%5.1f' "$(echo "($(date +%s)-$start)/60"|bc -l)")m] $*"; }

say "training until ${PATIENCE} epochs without improvement; approx \$${RATE_PER_HOUR}/h"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$WORK"
# Installs and downloads overlap: pip is network-bound and so is Kaggle.
pip install -q ultralytics kaggle pycocotools opencv-python-headless &
PIP_PID=$!

git clone --depth 1 --branch "$BRANCH" "$REPO_URL" Sol
wait $PIP_PID
say "deps installed, repo cloned"

mkdir -p ~/.kaggle
printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Competition data and the exp_002 checkpoint, fetched concurrently.
( kaggle competitions download -c "$COMP" -p "$WORK/data" -q && \
  unzip -q -o "$WORK/data/${COMP}.zip" -d "$WORK/data" && rm -f "$WORK/data/${COMP}.zip" ) &
DATA_PID=$!
( kaggle kernels output cosmicwitness/sol-exp002-yolo-seg -p "$WORK/ckpt" -q ) &
CKPT_PID=$!
wait $DATA_PID $CKPT_PID
say "data and checkpoint on disk"

ROOT=$(dirname "$(find "$WORK/data" -name 'MAGFiLO_1.0_Annotations_kaggle2026_train.json' | head -1)")
ROOT=$(dirname "$ROOT")
WEIGHTS=$(find "$WORK/ckpt" -name 'best.pt' | head -1)
say "data root: $ROOT"
say "weights:   $WEIGHTS"

cd "$WORK/Sol"
export PYTHONPATH="$WORK/Sol"

python -m experiments.exp_002_yolo_seg.src.prepare_yolo \
  --annotations "$ROOT/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json" \
  --images "$ROOT/train/train_images" --output "$WORK/yolo_ds"
say "dataset prepared"

python "$WORK/Sol/runpod/train.py" \
  --data "$WORK/yolo_ds/data.yaml" --weights "$WEIGHTS" \
  --out "$WORK/runs" --patience "$PATIENCE"
say "training finished"

BEST="$WORK/runs/ft2048/weights/best.pt"
[ -f "$BEST" ] || BEST="$WORK/runs/ft2048/weights/last.pt"

# Falsification test and operating-point search in one. If full-resolution mask
# supervision fixed the fat-mask problem, the optimal erosion moves from -1
# towards 0; if it stays at -1, the hypothesis was wrong whatever the score does.
python -m experiments.exp_005_postproc.src.nearmiss \
  --weights "$BEST" \
  --annotations "$ROOT/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json" \
  --images "$ROOT/train/train_images" --imgsz 2048 \
  --out "$WORK/nearmiss_after.json"
say "erosion sweep on the retrained model complete"

BEST_CONF=$(python -c "import json;print(json.load(open('$WORK/nearmiss_after.json'))['best']['conf'])")
BEST_GROW=$(python -c "import json;print(json.load(open('$WORK/nearmiss_after.json'))['best']['grow'])")
BEST_PQ=$(python -c "import json;print(round(json.load(open('$WORK/nearmiss_after.json'))['best']['pq'],4))")
say "optimum after retraining: conf $BEST_CONF grow $BEST_GROW (val PQ $BEST_PQ)"
say "  before retraining it was conf 0.35 grow -1 (val PQ 0.4404)"

python -m experiments.exp_002_yolo_seg.src.predict \
  --weights "$BEST" --images "$ROOT/test/test_images" \
  --output "$WORK/submission.csv" --imgsz 2048 \
  --conf "$BEST_CONF" --min-area 300 --grow "$BEST_GROW"
say "submission written"

kaggle competitions submit -c "$COMP" -f "$WORK/submission.csv" \
  -m "exp_010: @2048 retrained on 0.5px polygon-offset targets, conf $BEST_CONF grow $BEST_GROW (val PQ $BEST_PQ)"
say "SUBMITTED. Pod may be terminated."
