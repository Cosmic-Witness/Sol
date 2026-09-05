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

    # Each attempt in its own process. Version 2 looped inside one process and
    # two rungs reported byte-identical out-of-memory errors while the batch
    # between them halved -- the second never ran, because empty_cache() cannot
    # reclaim what the previous trainer still references. Rung 3 was then
    # measured against a heap polluted by both.
    #
    # What matters is the loss grid in native pixels, `2048 * mask_ratio /
    # imgsz`: the polygon correction is half a native pixel, representable at
    # 1.00, roughly so at 1.14, invisible at 2.00. So the ladder gives up
    # training resolution before supervision resolution, and never reaches
    # mask_ratio=2 at all.
    # Measured, in clean processes: the failing allocation is independent of
    # batch size (halving it changed 3.11 GiB by nothing) and linear in image
    # size (2048 -> 1792 moved it 3.11 -> 2.75, a factor of 0.88 against 0.875).
    # So batch is not a lever here and imgsz is the only supported one. Each rung
    # coarsens the loss grid; even the last is finer than the mask_ratio=2 that
    # made exp_010's central change inert at 2.00.
    attempts = ((2048, 2, 1.00), (1792, 2, 1.14), (1536, 2, 1.33), (1280, 4, 1.60))
    for imgsz, batch, grid in attempts:
        print(f"\n=== attempting imgsz {imgsz} batch {batch} "
              f"(loss grid {grid:.2f} native px) ===", flush=True)
        shutil.rmtree(RUNS_DIR / RUN_NAME, ignore_errors=True)
        result = subprocess.run([
            sys.executable, "-m", "experiments.exp_031_frozen.src.train",
            "--data", str(DATASET_DIR / "data.yaml"),
            "--weights", str(seeds[0]),
            "--project", str(RUNS_DIR), "--name", RUN_NAME,
            "--imgsz", str(imgsz), "--batch", str(batch),
            "--epochs", str(EPOCHS), "--freeze", str(FREEZE),
        ])
        if result.returncode == 0:
            print(f"\ntrained at imgsz {imgsz} batch {batch}", flush=True)
            break
        print(f"imgsz {imgsz} batch {batch} failed with {result.returncode}", flush=True)
    else:
        raise SystemExit(
            "full-resolution mask supervision does not fit at any resolution "
            "tried, each in a clean process. That is the result; do not retry "
            "at mask_ratio=2.")

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
