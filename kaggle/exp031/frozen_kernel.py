"""GPU kernel: the run exp_010 was supposed to be.

exp_010 intended three changes together — training at the inference resolution,
rasterisation-corrected targets, and full-resolution mask supervision. It got the
first two. `mask_ratio=1` did not fit on a T4 even at batch 1, the ladder fell
silently to 2, and at half resolution a half-pixel polygon correction is a
quarter of a pixel on the loss grid and does almost nothing. It also ran a cosine
schedule sized for 400 epochs and reached 62, so the learning rate never left its
peak and the model never consolidated.

Two changes fix both.

**Freeze the backbone.** Version 1 of this kernel claimed that would buy the
memory for full-resolution supervision. It did not: the first attempt still sat
at 13.2 GB and still failed, because the cost is not backbone activations but a
single 2.34 GB allocation in the loss, where 32 prototypes are upsampled to the
target grid. The freeze is kept anyway on its own merits -- this fine-tune is
correcting a systematic half-pixel bias in the masks, not relearning what a
filament looks like, and the decoder is where that correction lives -- but it is
not what makes the run fit.

**Fix the allocator, then trade image size for grid size.** The failure was 2.34
GB wanted against 2.06 GB free, with 984 MB reserved but unallocated: the
`expandable_segments` setting had been applied after `import torch` had already
initialised CUDA, so it never took effect. Setting it first may alone close a
280 MB gap. If it does not, the ladder drops the training resolution rather than
the supervision resolution, because the loss grid in native pixels is
`2048 * mask_ratio / imgsz` and 1792 at mask_ratio=1 is 1.14, nearly twice as
fine as 2048 at mask_ratio=2.

**Size the schedule to the session.** At roughly eight minutes an epoch with the
backbone frozen, seventy-five epochs fit inside the twelve-hour cap with margin,
so the cosine completes and the model gets the low-rate phase exp_010 never saw.

The memory ladder gives up batch before resolution and then **fails loudly**.
Falling back to `mask_ratio=2` is what made exp_010 measure the wrong thing for
twelve hours; if full resolution will not fit, that is the finding, and it should
arrive in minutes rather than being buried in a log.
"""

from __future__ import annotations

import os

# Before anything imports torch. The previous run set this after
# `torch.cuda.is_available()` had already initialised CUDA, so it did nothing --
# and the failure reported 984 MB reserved but unallocated, which is exactly what
# it would have reclaimed. Both spellings, since torch renamed the variable and
# the version here asks for the new one by name.
for _name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    os.environ.setdefault(_name, "expandable_segments:True")

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
RUNS_DIR = WORKING / "runs"
RUN_NAME = "frozen"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMGSZ = 2048
EPOCHS = 75
FREEZE = 11          # yolo11 backbone is layers 0-10; the neck and head train


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
    seeds = sorted(Path("/kaggle/input").rglob("weights/best.pt"))
    if not seeds:
        raise SystemExit("exp_010 checkpoint not attached")
    print(f"starting from {seeds[0]}", flush=True)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major * 10 + minor}"
    if capability not in torch.cuda.get_arch_list():
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

    run([sys.executable, "-m", "experiments.exp_002_yolo_seg.src.prepare_yolo",
         "--annotations", root / "train" / ANNOTATION_NAME,
         "--images", root / "train" / "train_images",
         "--output", DATASET_DIR])

    from ultralytics import YOLO

    def fit(imgsz: int, batch: int) -> None:
        YOLO(str(seeds[0])).train(
            data=str(DATASET_DIR / "data.yaml"),
            imgsz=imgsz, batch=batch, epochs=EPOCHS, patience=EPOCHS,
            project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True,
            freeze=FREEZE,
            optimizer="AdamW", lr0=5e-4, lrf=0.01, cos_lr=True, warmup_epochs=2.0,
            mask_ratio=1, overlap_mask=True,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
            fliplr=0.5, flipud=0.5, degrees=15.0, mosaic=0.0,
            cache=False, workers=2, seed=2026, verbose=True, plots=False,
        )

    # What matters is the loss grid measured in native pixels, which is
    # `2048 * mask_ratio / imgsz`. The polygon correction is half a native pixel,
    # so it is representable at 1.0 and roughly so at 1.14, and invisible at 2.0.
    # The ladder therefore gives up training resolution before it gives up
    # supervision resolution -- 1792 with mask_ratio=1 has a finer loss grid than
    # 2048 with mask_ratio=2, and costs only a 1.14x train/test mismatch against
    # the 1.6x exp_002 lived with.
    for imgsz, batch, grid in ((2048, 2, 1.00), (2048, 1, 1.00), (1792, 2, 1.14)):
        try:
            print(f"\nattempting mask_ratio=1 at imgsz {imgsz} batch {batch} "
                  f"(loss grid {grid:.2f} native px)", flush=True)
            fit(imgsz, batch)
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"imgsz {imgsz} batch {batch} did not fit: {exc}", flush=True)
            torch.cuda.empty_cache()
            shutil.rmtree(RUNS_DIR / RUN_NAME, ignore_errors=True)
    else:
        # Deliberately never mask_ratio=2. That is what exp_010 fell back to, and
        # it spent twelve hours measuring a configuration whose central change
        # was inert on the loss grid.
        raise SystemExit(
            "full-resolution mask supervision does not fit at any resolution "
            "tried. That is the result; do not retry at mask_ratio=2.")

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
