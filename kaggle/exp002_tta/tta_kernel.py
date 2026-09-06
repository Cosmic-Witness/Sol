"""Test-time augmentation for exp_002, on CPU so it costs no accelerator quota.

exp_002 scores 0.32 and both of its submissions were produced without TTA, which
was an oversight rather than a decision: predict.py has supported it throughout.
Averaging over flipped and rescaled views is a routine gain on segmentation and
costs nothing but time.

This kernel runs on CPU deliberately. Ultralytics infers perfectly well without a
GPU, 180 test images is a small job, and Kaggle's CPU kernels draw on neither the
GPU nor the TPU weekly pool. The accelerator quota is reserved for other work.

The validation sweep runs first and with the same TTA setting, so the confidence
threshold is chosen under the conditions the test predictions will actually use —
TTA changes the score distribution, and a threshold tuned without it would be
tuned for a different model.

Training and threshold selection are separated into two kernels on purpose.
Training is an eight-hour commitment; threshold selection is twenty minutes and
wants to be re-run whenever a new idea about post-processing shows up. Coupling
them would mean paying for the training every time.

The trained weights arrive by attaching the training kernel as a source, so this
kernel never retrains and the weights that produced a score stay traceable to the
run that made them.
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
IMGSZ = 1280


def run(command: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def find_data_root() -> Path:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        raise SystemExit(f"cannot find {ANNOTATION_NAME} under /kaggle/input")
    return matches[0].parent.parent


def find_weights() -> Path:
    """Locate best.pt inside whichever attached kernel output carries it."""
    candidates = sorted(Path("/kaggle/input").rglob("best.pt"))
    if not candidates:
        available = [str(p) for p in Path("/kaggle/input").glob("*")]
        raise SystemExit(f"no best.pt among attached sources. /kaggle/input holds: {available}")
    print(f"weights: {candidates[0]}", flush=True)
    return candidates[0]


def main() -> None:
    data_root = find_data_root()
    weights = find_weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch

    # CPU by design; see the module docstring. Threads are set explicitly because
    # Ultralytics otherwise leaves torch on a single thread for CPU inference.
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"torch {torch.__version__} on CPU, {os.cpu_count()} threads", flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["YOLO_CONFIG_DIR"] = str(SCRATCH / "ultralytics")
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    sweep_path = OUT_DIR / "sweep.json"
    run([
        sys.executable, "-m", "experiments.exp_002_yolo_seg.src.tune",
        "--weights", weights,
        "--annotations", data_root / "train" / ANNOTATION_NAME,
        "--images", data_root / "train" / "train_images",
        "--imgsz", IMGSZ,
        "--tta",
        "--out", sweep_path,
    ])

    best = json.loads(sweep_path.read_text())["best"]
    print(f"\nusing conf={best['conf']} min_area={best['min_area']} "
          f"(validation PQ {best['pq']:.4f})", flush=True)

    run([
        sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
        "--weights", weights,
        "--images", data_root / "test" / "test_images",
        "--output", OUT_DIR / "submission.csv",
        "--imgsz", IMGSZ,
        "--conf", best["conf"],
        "--min-area", best["min_area"],
        "--tta",
    ])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
