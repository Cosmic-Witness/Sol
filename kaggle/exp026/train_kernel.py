"""TPU kernel: train the candidate verifier."""

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


def run(command: list) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command])
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def main() -> None:
    crops = sorted(Path("/kaggle/input").rglob("train_label.npy"))
    if not crops:
        raise SystemExit("crops not attached")
    data = crops[0].parent
    print(f"crops {data}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([sys.executable, "-m", "experiments.exp_025_verifier.src.train_verifier",
         "--data", data, "--out", CKPT_DIR,
         "--epochs", 300, "--batch", 256, "--patience", 40])


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
