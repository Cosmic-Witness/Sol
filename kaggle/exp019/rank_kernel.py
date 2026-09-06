"""CPU kernel: can the mask field's fall-off promote candidates confidence discards?

Consumes exp_017's cached cuts, so there is no inference here.
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
    caches = sorted(Path("/kaggle/input").rglob("level_candidates.json"))
    if not caches:
        raise SystemExit("exp_017 cache not attached")
    print(f"data {root}\ncache {caches[0]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q",
         "pycocotools", "scikit-learn"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    # The levels come from the module that wrote the cache, so the two cannot
    # drift apart silently.
    sys.path.insert(0, str(REPO_DIR))
    from experiments.exp_017_softmask.src.logit_sweep import LOGIT_LEVELS

    run([sys.executable, "-m", "experiments.exp_019_fieldrank.src.rank",
         "--candidates", caches[0],
         "--levels", json.dumps(LOGIT_LEVELS),
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--out", OUT_DIR / "fieldrank.json"])


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
