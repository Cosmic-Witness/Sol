"""Kaggle TPU kernel for exp_004: high-resolution dense segmentation.

Why this is on TPU and dense rather than GPU and YOLO
-----------------------------------------------------
Ultralytics has no torch_xla support at all, so the YOLO line cannot run here.
The ablation in spine_ablation.py showed that is less costly than it sounds:
connected components recover ground-truth instances at PQ 1.000 when the mask is
accurate, so the decomposition was never what capped exp_001. Mask precision is,
and precision comes from resolution, capacity and epochs.

Checkpoints are written under /kaggle/working every epoch. exp_003 lost 1.5 h of
training because its checkpoints sat on /kaggle/temp, which is not part of the
kernel output; this run is recoverable at whatever epoch it reaches.
"""

from __future__ import annotations

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
CACHE_DIR = SCRATCH / "cache1024"

CKPT_DIR = WORKING / "checkpoints"
OUT_DIR = WORKING / "outputs"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

SIZE = 1024
ENCODER = "resnet34"
EPOCHS = 300
BATCH = 16              # global batch across 8 cores; must divide by device count
TIME_BUDGET_HOURS = 7.0


def run(command: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def find_data_root() -> Path:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        available = [str(p) for p in Path("/kaggle/input").glob("*")]
        raise SystemExit(f"cannot find {ANNOTATION_NAME}. /kaggle/input holds: {available}")
    root = matches[0].parent.parent
    print(f"data root: {root}", flush=True)
    return root


def report_environment() -> None:
    """Report the environment WITHOUT touching the TPU.

    The TPU is a single vfio device that exactly one process may hold. Calling
    xm.xla_device() here claimed it for this driver, and the training subprocess
    then died with "open(/dev/vfio/0): Device or resource busy" followed by
    "InitializeComputationClient() can only be called once". Importing
    torch_xla at all in the parent risks initialising the runtime, so the parent
    reports only what it can see without it and leaves the device untouched for
    the trainer.
    """
    print("=" * 70, flush=True)
    import torch

    print(f"torch {torch.__version__}", flush=True)
    for key in ("PJRT_DEVICE", "TPU_ACCELERATOR_TYPE", "TPU_WORKER_ID"):
        print(f"{key}={os.environ.get(key)}", flush=True)
    if os.environ.get("PJRT_DEVICE") != "TPU":
        raise SystemExit("no TPU runtime: PJRT_DEVICE is not TPU")
    print("TPU left uninitialised for the training process", flush=True)
    print("=" * 70, flush=True)


def main() -> None:
    data_root = find_data_root()
    report_environment()
    for directory in (CKPT_DIR, OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "-q",
         "segmentation-models-pytorch", "opencv-python-headless", "pycocotools"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)
    os.environ["PJRT_DEVICE"] = "TPU"

    run([sys.executable, "-m", "experiments.exp_004_spine_tpu.src.prepare",
         "--annotations", data_root / "train" / ANNOTATION_NAME,
         "--images", data_root / "train" / "train_images",
         "--output", CACHE_DIR, "--size", SIZE])

    run([sys.executable, "-m", "experiments.exp_004_spine_tpu.src.train_xla",
         "--cache", CACHE_DIR, "--out", CKPT_DIR, "--encoder", ENCODER,
         "--epochs", EPOCHS, "--batch", BATCH, "--time-budget", TIME_BUDGET_HOURS])

    best = CKPT_DIR / "best.pt"
    if not best.exists():
        raise SystemExit("training produced no checkpoint")

    run([sys.executable, "-m", "experiments.exp_004_spine_tpu.src.predict",
         "--checkpoint", best, "--images", data_root / "test" / "test_images",
         "--output", OUT_DIR / "submission.csv", "--encoder", ENCODER,
         "--size", SIZE, "--tta"])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
