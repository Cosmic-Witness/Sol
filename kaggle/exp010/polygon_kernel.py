"""GPU kernel: retrain at the inference resolution, on correctly rasterised targets.

Two defects in the shipped detector, both in training rather than in inference.

The first is the targets. Ultralytics rasterises polygons with `cv2.fillPoly`
and the scorer rasterises them with pycocotools; the conventions differ, and
measured over 250 instances the training mask is 11% fatter than the mask the
score is computed against (IoU 0.898). A half-pixel inward polygon buffer
reconciles them to IoU 0.959. `prepare_yolo.py` applies it by default.

The second is resolution. exp_002 trained at 1280 and every submission since has
inferred at 2048, because inference at native resolution is worth 0.033 PQ over
1280. The model's learned prior over object sizes was set at 1280 and is being
asked about objects 1.6 times larger, which is the most plausible reading of why
2560 and 3072 degrade so sharply: the head responds over a limited size range,
and the range was fixed by the training resolution.

With the targets correct, full-resolution mask supervision becomes worth asking
for again. Two thirds of the model's error sits within two pixels of the
boundary, and the default `mask_ratio=4` computes the mask loss on a 512 grid
where that entire band is invisible. It was withdrawn before on the grounds that
finer supervision against an 11%-fat target only reproduces the fat target more
faithfully -- true then, and no longer true.

exp_002 also ran under a clock and stopped at 149 epochs with the time budget,
not with convergence or patience. This run has no clock. It writes checkpoints
into the kernel output, so the 12-hour cap is a checkpoint rather than a
deadline, and a later version of this kernel that lists its own output as a
source resumes with the optimiser state, the learning-rate schedule, the epoch
counter and the best-so-far fitness all intact.

Prediction lives in a separate kernel (exp_015) so that training is never cut
short to leave room for it, and so a submission can be taken from whatever
checkpoint exists at any point.
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
DATASET_DIR = SCRATCH / "yolo_ds"
RUNS_DIR = WORKING / "runs"          # kernel output: survives interruption
RUN_NAME = "polygon"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMGSZ = 2048
PATIENCE = 60


def run(command: list) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command])
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def restore_previous_run() -> bool:
    """Copy a previous session's run directory back into place.

    Ultralytics resumes from a run *directory*, not from a checkpoint alone: it
    needs `args.yaml` and `results.csv` beside the weights to continue the epoch
    counter and the schedule. Starting a fresh run from `last.pt` instead would
    reset `best_fitness` to zero, and the first epoch of the new session would
    overwrite a `best.pt` it has not yet beaten.
    """
    previous = sorted(Path("/kaggle/input").rglob(f"runs/{RUN_NAME}/weights/last.pt"))
    if not previous:
        return False
    source = previous[0].parent.parent
    destination = RUNS_DIR / RUN_NAME
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"restored previous run from {source}", flush=True)
    return True


def main() -> None:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        raise SystemExit("competition data not attached")
    root = matches[0].parent.parent

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major * 10 + minor}"
    if capability not in torch.cuda.get_arch_list():
        # A P100 reports itself available and then fails in the first backward
        # pass, half an hour in. Fail here instead.
        raise SystemExit(f"torch has no kernels for {capability}; request a T4")
    print(f"device {torch.cuda.get_device_name(0)} ({capability})", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
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

    resuming = restore_previous_run()
    if resuming:
        model = YOLO(str(RUNS_DIR / RUN_NAME / "weights" / "last.pt"))
        model.train(resume=True)
        print("\nDONE (resumed session).", flush=True)
        return

    seeds = sorted(Path("/kaggle/input").rglob("checkpoints/best.pt"))
    if not seeds:
        raise SystemExit("no exp_002 checkpoint to start from")
    print(f"starting from exp_002 {seeds[0]}", flush=True)

    def fit(mask_ratio: int) -> None:
        model = YOLO(str(seeds[0]))
        model.train(
            data=str(DATASET_DIR / "data.yaml"),
            imgsz=IMGSZ, batch=2, epochs=400, patience=PATIENCE,
            project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True,
            lr0=0.0005, warmup_epochs=1.0,
            mask_ratio=mask_ratio, overlap_mask=True,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
            fliplr=0.5, flipud=0.5, degrees=15.0, mosaic=0.0,
            cache=False, workers=2, seed=2026, verbose=True, plots=False,
        )

    # Upsampling the 32 prototypes to 2048 square inside the loss costs about a
    # gigabyte under autocast. If the T4 cannot hold it, half resolution still
    # doubles what the default sees.
    try:
        fit(1)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"\nmask_ratio=1 did not fit ({exc}); falling back to 2", flush=True)
        torch.cuda.empty_cache()
        shutil.rmtree(RUNS_DIR / RUN_NAME, ignore_errors=True)
        fit(2)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
