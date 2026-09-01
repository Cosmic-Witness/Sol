"""Does exp_002 score better when inferred at a higher resolution than it trained at?

The measured deficit is recall: at 1280, about 40% of ground-truth filaments are
never proposed at any confidence threshold, and barbs are a few pixels wide at
2048. exp_003 was built to address that by retraining at full resolution, which
needs GPU. Inference resolution is a free proxy for the same question — a
detector run on a larger input sees the same structures at more pixels, and for
thin objects that alone sometimes recovers detections.

It can equally hurt: a model evaluated far from its training scale can degrade,
and YOLO's anchor-free heads are only somewhat scale-robust. That is why this
sweeps validation rather than guessing and submitting.

Runs on CPU. Ultralytics infers without a GPU, and Kaggle CPU kernels draw on
neither accelerator pool, so this costs none of the reserved quota.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Cosmic-Witness/Sol"
BRANCH = "claude/kaggle-credentials-setup-f7nudy"

WORKING = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp")
REPO_DIR = SCRATCH / "Sol"
OUT_DIR = WORKING / "outputs"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RESOLUTIONS = (1280, 1600, 2048)


def run(command: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def main() -> None:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        raise SystemExit("competition data not attached")
    data_root = matches[0].parent.parent

    weights = sorted(Path("/kaggle/input").rglob("best.pt"))
    if not weights:
        raise SystemExit("exp_002 checkpoint not attached")
    print(f"weights: {weights[0]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"torch {torch.__version__} on CPU, {os.cpu_count()} threads", flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    summary = {}
    for imgsz in RESOLUTIONS:
        sweep = OUT_DIR / f"sweep_{imgsz}.json"
        print(f"\n{'=' * 60}\nvalidation sweep at imgsz {imgsz}\n{'=' * 60}", flush=True)
        run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.tune",
             "--weights", weights[0],
             "--annotations", data_root / "train" / ANNOTATION_NAME,
             "--images", data_root / "train" / "train_images",
             "--imgsz", imgsz, "--out", sweep])
        summary[imgsz] = json.loads(sweep.read_text())["best"]

    print(f"\n{'=' * 60}\nRESOLUTION COMPARISON (validation PQ)\n{'=' * 60}", flush=True)
    for imgsz, best in summary.items():
        print(f"  imgsz {imgsz}: PQ {best['pq']:.4f} "
              f"(conf {best['conf']}, min_area {best['min_area']}, "
              f"TP {best['tp']} FP {best['fp']} FN {best['fn']})", flush=True)

    winner = max(summary.items(), key=lambda kv: kv[1]["pq"])
    imgsz, best = winner
    baseline = summary[1280]["pq"]
    print(f"\nbest: imgsz {imgsz} at PQ {best['pq']:.4f} "
          f"({best['pq'] - baseline:+.4f} against 1280)", flush=True)
    (OUT_DIR / "resolution_summary.json").write_text(json.dumps(summary, indent=2))

    # Only produce a submission if a higher resolution actually won; otherwise the
    # existing 1280 submission already represents the best this model can do.
    if imgsz == 1280:
        print("1280 remains best; no new submission generated.", flush=True)
        return

    run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
         "--weights", weights[0],
         "--images", data_root / "test" / "test_images",
         "--output", OUT_DIR / "submission.csv",
         "--imgsz", imgsz, "--conf", best["conf"], "--min-area", best["min_area"]])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
