"""Retrain at 2048 on rasterisation-corrected targets, where interruption is survivable.

The experiment
--------------
Every detector on this project has been fitted to targets 10.8% larger in area
than the scorer measures: Ultralytics rasterises with cv2.fillPoly, the
competition scores with pycocotools, and a filament is nearly all perimeter.
Measured agreement is IoU 0.898. A 0.5px inward polygon offset — applied in
coordinate space, so with no quantisation floor — closes it to 0.959 and moves
the area ratio from 1.111 to 0.986.

Why here rather than on rented hardware
---------------------------------------
The same run on RunPod cost $2.39 and produced nothing. Training was given no
clock, correctly, but the pod had no persistent volume and a hard spend limit
killed it mid-training, so every checkpoint died with the container and the
predict-and-submit steps never ran.

The precondition for "no time budget" is that an interrupted run keeps its work.
That holds here and did not hold there: RUNS_DIR is under /kaggle/working, which
*is* the kernel output, so a session killed at the 12-hour cap still yields
best.pt and last.pt. The lesson is not "always set a clock" but "only remove the
clock where the work survives without one".

Resuming
--------
If a previous run of this kernel is attached as a source, training continues from
its last.pt rather than restarting. That makes the 12-hour cap a checkpoint
rather than a wall, and several sessions can be chained on free quota to reach a
budget no single paid run could.
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
DATASET_DIR = SCRATCH / "yolo_ds"
RUNS_DIR = WORKING / "runs"          # kernel output: survives interruption
OUT_DIR = WORKING / "outputs"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMGSZ = 2048
PATIENCE = 40


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

    # Prefer a previous run of this kernel (resume); fall back to exp_002.
    resumes = sorted(Path("/kaggle/input").rglob("runs/*/weights/last.pt"))
    if resumes:
        start = resumes[0]
        print(f"resuming from {start}", flush=True)
    else:
        seeds = sorted(Path("/kaggle/input").rglob("best.pt"))
        if not seeds:
            raise SystemExit("no checkpoint to start from")
        start = seeds[0]
        print(f"starting from exp_002 {start}", flush=True)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    major, minor = torch.cuda.get_device_capability()
    capability = f"sm_{major * 10 + minor}"
    if capability not in torch.cuda.get_arch_list():
        raise SystemExit(f"torch has no kernels for {capability}; request a T4")
    print(f"device {torch.cuda.get_device_name(0)} ({capability})", flush=True)

    for d in (RUNS_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "shapely", "pycocotools"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])
    os.environ["PYTHONPATH"] = str(REPO_DIR)
    os.environ["YOLO_CONFIG_DIR"] = str(SCRATCH / "ultralytics")

    # prepare_yolo applies the 0.5px inward offset by default.
    run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.prepare_yolo",
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--output", DATASET_DIR])

    from ultralytics import YOLO

    def fit(mask_ratio: int) -> None:
        model = YOLO(str(start))
        model.train(
            data=str(DATASET_DIR / "data.yaml"),
            imgsz=IMGSZ, batch=2, epochs=1000, patience=PATIENCE,
            project=str(RUNS_DIR), name="polygon", exist_ok=True,
            lr0=0.0005, warmup_epochs=1.0,
            mask_ratio=mask_ratio, overlap_mask=True,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
            fliplr=0.5, flipud=0.5, degrees=15.0, mosaic=0.0,
            cache=False, workers=2, seed=2026, verbose=True,
        )

    # `mask_ratio=1` supervises the mask loss at the full 2048 grid instead of
    # the default 512. Two thirds of the model's error sits within two pixels of
    # the boundary, and at a quarter resolution that entire band is invisible to
    # the loss. It is only worth asking for now that the targets themselves are
    # correct -- against an 11%-fat target, finer supervision reproduces the fat
    # target more faithfully, which is why this was withdrawn before.
    #
    # The cost is upsampling the 32 prototypes to 2048 square inside the loss,
    # about a gigabyte under autocast. If the T4 cannot hold it, half resolution
    # still doubles what the default sees.
    try:
        fit(1)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"\nmask_ratio=1 did not fit ({exc}); falling back to 2", flush=True)
        torch.cuda.empty_cache()
        fit(2)

    best = RUNS_DIR / "polygon" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit("no checkpoint produced")

    # The shipped operating point (conf 0.35, one-pixel erosion) was tuned for a
    # detector trained on fat targets. This detector should not need the erosion
    # at all, but that is a prediction, not a measurement -- so measure it.
    sweep_path = OUT_DIR / "nearmiss.json"
    run([sys.executable, "-m", "experiments.exp_005_postproc.src.nearmiss",
         "--weights", best,
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--imgsz", IMGSZ, "--out", sweep_path,
         "--dump-cache", OUT_DIR / "candidates.json"])

    sweep = json.loads(Path(sweep_path).read_text())["sweep"]
    top = max(sweep, key=lambda row: row["pq"])
    print(f"\nbest validation point: {json.dumps(top, indent=2)}", flush=True)

    run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
         "--weights", best, "--images", root / "test" / "test_images",
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
