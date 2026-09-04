"""CPU kernel: harvest real detector-versus-truth pairs.

GPU quota is exhausted until the weekly reset, so this runs on CPU. Detector
inference over the training photographs at 2048 is the expensive part; the
resulting pairs are small and go into the kernel output, where the refiner
training can consume them without repeating the inference.

Restricted to a subset of photographs so the run fits inside the 12-hour cap
with margin. About 300 photographs yields several thousand pairs, which is ample
for a 1.9M-parameter refiner and far more informative than the 20571 synthetic
pairs the first attempt used.
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
OUT_DIR = WORKING / "pairs"
ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
LIMIT = 300


def run(command: list) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command])
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def main() -> None:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        raise SystemExit("competition data not attached")
    root = matches[0].parent.parent
    weights = sorted(Path("/kaggle/input").rglob("best.pt"))
    if not weights:
        raise SystemExit("detector checkpoint not attached")
    print(f"data {root}\nweights {weights[0]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([sys.executable, "-m", "experiments.exp_008_refiner.src.harvest",
         "--weights", weights[0],
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--output", OUT_DIR, "--imgsz", 2048, "--limit", LIMIT])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
