"""CPU kernel: decompose the validation error into its failure classes.

Consumes the candidate cache exp_005 wrote, so no detector inference happens
here. The run is a few minutes of RLE arithmetic and costs no GPU quota.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Cosmic-Witness/Sol"
BRANCH = "claude/kaggle-credentials-setup-f7nudy"
SCRATCH = Path("/kaggle/temp")
REPO_DIR = SCRATCH / "Sol"
WORKING = Path("/kaggle/working")
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
    caches = sorted(Path("/kaggle/input").rglob("candidates.json"))
    if not caches:
        raise SystemExit("candidate cache not attached")
    print(f"data {root}\ncache {caches[0]}", flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    out = WORKING / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "experiments.exp_013_errors.src.recall_ceiling",
         "--candidates", caches[0],
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--out", out / "recall_ceiling.json"])


if __name__ == "__main__":
    main()
