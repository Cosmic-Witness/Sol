"""GPU kernel: find the 2048-trained model's best inference resolution."""

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
OUT_DIR = WORKING / "outputs"
ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


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
    weights = sorted(Path("/kaggle/input").rglob("weights/best.pt"))
    if not weights:
        raise SystemExit("exp_010 checkpoint not attached")
    print(f"weights {weights[0]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)
    os.environ["YOLO_CONFIG_DIR"] = str(SCRATCH / "ultralytics")

    run([sys.executable, "-m", "experiments.exp_024_infres.src.scale",
         "--weights", weights[0],
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--out", OUT_DIR / "infres.json"])


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
