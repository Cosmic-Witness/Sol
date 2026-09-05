"""CPU kernel: pick an operating point on validation, then predict the test set.

Split out of the training kernel so that training never has to stop early to
leave room for inference, and so a submission can be taken from whatever
checkpoint exists at the time.

CPU rather than GPU on purpose. Inference over 106 validation and 180 test
photographs at 2048 is roughly ninety minutes on Kaggle's CPU and nothing on a
T4, but GPU hours are the scarce resource and every one of them belongs in the
training kernel. The 12-hour CPU cap is ample.

The operating point is measured, not inherited. conf 0.35 with a one-pixel
erosion was tuned against a detector trained on targets 11% too fat; a detector
trained on corrected targets should want no erosion at all, which is a
prediction rather than a measurement.
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
IMGSZ = 2048


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

    weights = sorted(Path("/kaggle/input").rglob("runs/*/weights/best.pt"))
    if not weights:
        weights = sorted(Path("/kaggle/input").rglob("checkpoints/best.pt"))
    if not weights:
        # Worth knowing exactly what did mount. exp_010 is cancelled at the
        # twelve-hour cap rather than completing, and whether Kaggle publishes a
        # cancelled kernel's output as an attachable source decides whether the
        # three-session resume plan works at all -- the self-reference probe only
        # ever tested a kernel that finished.
        print("=== /kaggle/input ===", flush=True)
        for path in sorted(Path("/kaggle/input").rglob("*"))[:60]:
            print(f"  {path}", flush=True)
        raise SystemExit("no checkpoint attached")
    print(f"data {root}\nweights {weights[0]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)
    os.environ["YOLO_CONFIG_DIR"] = str(SCRATCH / "ultralytics")

    sweep_path = OUT_DIR / "nearmiss.json"
    run([sys.executable, "-m", "experiments.exp_005_postproc.src.nearmiss",
         "--weights", weights[0],
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--imgsz", IMGSZ, "--out", sweep_path,
         "--dump-cache", OUT_DIR / "candidates.json"])

    sweep = json.loads(sweep_path.read_text())["sweep"]
    top = max(sweep, key=lambda row: row["pq"])
    print(f"\nbest validation point: {json.dumps(top, indent=2)}", flush=True)

    run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
         "--weights", weights[0], "--images", root / "test" / "test_images",
         "--output", OUT_DIR / "submission.csv",
         "--imgsz", IMGSZ,
         "--conf", top["conf"], "--min-area", top["min_area"],
         "--grow", top["grow"]])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
