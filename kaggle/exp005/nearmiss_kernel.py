"""CPU kernel: sweep mask growth against the near-miss population on validation.

Costs neither accelerator pool. GPU quota is exhausted until the weekly reset and
Ultralytics has no XLA backend, so CPU inference on the existing exp_002 weights
is the only compute available — and the near-miss arithmetic says it is where the
cheapest true positives are.
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

    weights = sorted(Path("/kaggle/input").rglob("best.pt"))
    if not weights:
        raise SystemExit("exp_002 checkpoint not attached")
    print(f"data {root}\nweights {weights[0]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([sys.executable, "-m", "experiments.exp_005_postproc.src.nearmiss",
         "--weights", weights[0],
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--imgsz", 2048,
         "--out", OUT_DIR / "nearmiss.json",
         "--dump-cache", SCRATCH / "candidates.json"])

    # Calibration reuses the cached candidates; it never re-runs the model.
    run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    run([sys.executable, "-m", "experiments.exp_005_postproc.src.calibrate",
         "--cache", SCRATCH / "candidates.json",
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--out", OUT_DIR / "calibrate.json",
         "--grow", -1, "--min-area", 300])

    # Apply the winning setting to the test set in the same kernel: the sweep has
    # already paid for loading the model, and a separate run would repeat it.
    best = json.loads((OUT_DIR / "nearmiss.json").read_text())["best"]
    print(f"\napplying conf={best['conf']} grow={best['grow']} "
          f"(validation PQ {best['pq']:.4f})", flush=True)
    run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
         "--weights", weights[0],
         "--images", root / "test" / "test_images",
         "--output", OUT_DIR / "submission.csv",
         "--imgsz", 2048,
         "--conf", best["conf"],
         "--min-area", best["min_area"],
         "--grow", best["grow"]])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
