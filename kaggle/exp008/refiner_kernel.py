"""TPU kernel: train the boundary refiner.

The measurement says 64% of the model's mask error sits within two pixels of the
boundary and that eliminating it is worth about +0.11 PQ, while the global 1px
erosion currently in the pipeline recovers 0.023 of that. A per-pixel boundary
model is what captures the rest.

This runs on TPU rather than GPU because GPU quota is exhausted until the weekly
reset and the refiner is a plain PyTorch U-Net, which torch_xla handles — the
same SPMD arrangement exp_004 used to train 300 epochs in 3.8 hours. Ultralytics
could not use the TPU; this can.

Checkpoints land in /kaggle/working so an interrupted session keeps its epochs,
and the parent process never touches the TPU, both lessons from exp_004.
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
CROPS_DIR = SCRATCH / "crops"
CKPT_DIR = WORKING / "checkpoints"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
EPOCHS = 600              # a ceiling, not a target; early stopping decides
BATCH = 64                # global across 8 cores


def run(command: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def main() -> None:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        raise SystemExit("competition data not attached")
    root = matches[0].parent.parent
    print(f"data root: {root}", flush=True)

    # Report without initialising the TPU: it is a single vfio device that one
    # process may hold, and claiming it here would deny it to the trainer.
    print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE')} "
          f"TPU_ACCELERATOR_TYPE={os.environ.get('TPU_ACCELERATOR_TYPE')}", flush=True)
    if os.environ.get("PJRT_DEVICE") != "TPU":
        raise SystemExit("no TPU runtime")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "pycocotools", "opencv-python-headless"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([sys.executable, "-m", "experiments.exp_008_refiner.src.crops",
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--output", CROPS_DIR, "--per-instance", 3])

    run([sys.executable, "-m", "experiments.exp_008_refiner.src.train_refiner",
         "--data", CROPS_DIR, "--out", CKPT_DIR,
         "--epochs", EPOCHS, "--batch", BATCH])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
