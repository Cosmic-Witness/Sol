"""Measure the real-pair refiner on real detector output, then submit if it wins.

The refiner learned from synthetic degradation — ground truth dilated, drifted
and smoothed to imitate the detector's measured error modes. It reaches val IoU
0.8529 from a coarse input of 0.7208 on that task, and the simulation is
calibrated (0.72 against the detector's real SQ of 0.679).

None of which proves it helps on real masks. Synthetic degradation is a model of
the error, and a model can be wrong in ways that make the refiner learn a
correction the true errors do not need. So validation PQ is measured both ways
before a submission slot is spent, and the run reports mean IoU between refined
and coarse masks — near 1.0 would mean the stage is a no-op whatever the PQ says.
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

    # The detector and the refiner are both called best.pt, in different kernel
    # outputs; pick them apart by which kernel they came from.
    checkpoints = sorted(Path("/kaggle/input").rglob("best.pt"))
    detector = next((p for p in checkpoints if "yolo-seg" in str(p)), None)
    refiner = next((p for p in checkpoints if "refiner" in str(p)), None)
    if detector is None or refiner is None:
        raise SystemExit(f"need both checkpoints; found {[str(p) for p in checkpoints]}")
    print(f"detector {detector}\nrefiner  {refiner}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([sys.executable, "-m", "experiments.exp_008_refiner.src.evaluate",
         "--detector", detector, "--refiner", refiner,
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--out", OUT_DIR / "refiner_val.json"])

    import json
    verdict = json.loads((OUT_DIR / "refiner_val.json").read_text())
    print(f"\nbaseline PQ {verdict['baseline']['pq']:.4f} | "
          f"refined PQ {verdict['refined']['pq']:.4f}", flush=True)

    if verdict["refined"]["pq"] <= verdict["baseline"]["pq"]:
        # exp_009 forced a submission through this guard to buy a transfer
        # datum, and it bought one: validation 0.4322 became public 0.35 against
        # the baseline's 0.4404 and 0.36. Direction transferred. There is nothing
        # further to learn from spending a submission on a known regression, so
        # this run respects the guard.
        print("the refiner does not beat the baseline on validation; "
              "no submission written", flush=True)
        return

    run([sys.executable, "-m", "experiments.exp_008_refiner.src.apply",
         "--weights", detector, "--refiner", refiner,
         "--images", root / "test" / "test_images",
         "--output", OUT_DIR / "submission.csv",
         "--imgsz", 2048, "--conf", 0.35, "--min-area", 300,
         "--threshold", verdict["best_threshold"],
         # -1 erodes the refined mask; the evaluation decides whether that wins.
         "--post-grow", -1 if verdict.get("best_erode") else 0])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
