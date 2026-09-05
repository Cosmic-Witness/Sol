"""TPU kernel: retrain the refiner on real detector-versus-truth pairs.

The first refiner learned from ground truth I had dilated and blurred myself. It
lost to a 1px erosion. The harvest explains why: real coarse masks sit at IoU
0.6105 against their truth, while my synthetic ones sat at 0.7258. The damage was
too mild by 0.115.

The calibration error is worth stating precisely, because it looked rigorous.
Severity was matched against the detector's SQ of 0.679 — but SQ is the mean IoU
over *matched* pairs only, those already above the 0.5 threshold. It is a biased
statistic for this purpose, and the population the refiner must actually repair
includes every near-miss down to 0.25. Calibrating against the matched subset
guaranteed the simulation was easier than reality.

These pairs are the detector's real output. No simulation, and no possibility of
the same error.
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
CKPT_DIR = WORKING / "checkpoints"
EPOCHS = 600            # ceiling; patience decides
BATCH = 32              # global across 8 cores; 3429 pairs is a smaller set
PATIENCE = 40


def run(command: list) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command])
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def main() -> None:
    pairs = sorted(Path("/kaggle/input").rglob("train_truth.npy"))
    if not pairs:
        raise SystemExit("harvested pairs not attached")
    data_dir = pairs[0].parent
    print(f"pairs: {data_dir}", flush=True)

    print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE')}", flush=True)
    if os.environ.get("PJRT_DEVICE") != "TPU":
        raise SystemExit("no TPU runtime")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)
    os.environ["PJRT_DEVICE"] = "TPU"

    run([sys.executable, "-m", "experiments.exp_008_refiner.src.train_refiner",
         "--data", data_dir, "--out", CKPT_DIR,
         "--epochs", EPOCHS, "--batch", BATCH, "--patience", PATIENCE])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
