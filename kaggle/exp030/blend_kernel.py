"""CPU kernel: fuse the 1280-trained and 2048-trained detectors.

Both candidate sets are already cached in kernel outputs, so this is RLE
arithmetic and needs no inference and no GPU.
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

    caches = sorted(Path("/kaggle/input").rglob("candidates.json"))
    a = next((p for p in caches if "exp005" in str(p)), None)
    b = next((p for p in caches if "exp015" in str(p)), None)
    if a is None or b is None:
        print("=== caches found ===", flush=True)
        for p in caches:
            print(f"  {p}", flush=True)
        raise SystemExit("need both candidate caches")
    print(f"A (exp_002, via exp_005) {a}\nB (exp_010, via exp_015) {b}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([sys.executable, "-m", "experiments.exp_021_ensemble.src.blend",
         "--candidates-a", a, "--candidates-b", b,
         
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--out", OUT_DIR / "blend.json"])


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
