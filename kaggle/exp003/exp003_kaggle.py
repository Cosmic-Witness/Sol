"""Kaggle kernel driver for experiment 003: large YOLO at full resolution.

Why this run exists
-------------------
exp_002 was configured at yolo11m-seg / 1280 to fit comfortably inside one T4
session. That was a memory constraint allowed to masquerade as a modelling
decision, and it was the wrong trade. Filament barbs are a few pixels wide at
2048, recall on thin structures is bought almost entirely with resolution, and
the strongest public solution on this leaderboard is a *large* segmentation
backbone at *2048*. Training smaller and shorter than the known-good reference
guarantees landing under it.

This run matches the reference envelope instead of hedging against it:

    yolo11l-seg, imgsz 2048, batch chosen by autobatch, ~11 h of training

Memory is handled properly rather than by shrinking the problem. `batch=-1` asks
Ultralytics to measure free VRAM and pick the largest batch that fits, and the
nominal-batch accumulation keeps the effective batch at 64 regardless of what
that turns out to be, so gradient statistics do not degrade when the physical
batch is small. If two T4s are present, both are used.
"""

from __future__ import annotations

import os
import shutil
import time
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Cosmic-Witness/Sol"
BRANCH = "claude/kaggle-credentials-setup-f7nudy"

WORKING = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp")
REPO_DIR = SCRATCH / "Sol"
DATASET_DIR = SCRATCH / "yolo_dataset"
RUNS_DIR = SCRATCH / "runs"

CKPT_DIR = WORKING / "checkpoints"
OUT_DIR = WORKING / "outputs"

ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

# Progressive resizing: exp_003 continues exp_002 rather than restarting.
# exp_002 reached its best mask mAP50 at epoch 149 of 149 and was still
# improving when the clock stopped it, so its weights are a far better starting
# point than COCO pretraining. Starting fresh at 2048 would also buy fewer
# epochs per hour, so a from-scratch run would be compared against exp_002 while
# undertrained relative to it, and the resolution question would go unanswered.
# FALLBACK_MODEL is used only if no exp_002 checkpoint is attached.
FALLBACK_MODEL = "yolo11m-seg.pt"
IMGSZ = 2048
EPOCHS = 300          # never reached; the time budget is the real stop condition
# 9.38 h of weekly quota remain. 7.0 h of training leaves room for dataset
# preparation, prediction over 180 images at 2048, and the output upload,
# without risking the run being killed by quota exhaustion mid-epoch.
TIME_BUDGET_HOURS = 7.0

# Batch sizes to try, largest first. The first attempt errored with
#   "AutoBatch with batch<1 not supported for Multi-GPU training"
# because the session provides two T4s and batch=-1 cannot be split across them.
# An explicit batch is required, and it must be a multiple of the GPU count.
#
# Rather than guess once and lose the session to an out-of-memory error hours in,
# the ladder is tried in order and a memory failure falls through to the next
# rung. 4 is the estimate: a 1280 run holds roughly batch 8 on one T4, and 2048
# carries (2048/1280)^2 = 2.56x the pixels, so ~3 per GPU is the expected limit.
BATCH_LADDER = (8, 4, 2)


def run(command: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def find_start_weights() -> str:
    """Prefer exp_002's checkpoint; fall back to COCO pretraining."""
    candidates = sorted(Path("/kaggle/input").rglob("best.pt"))
    if candidates:
        print(f"resuming from {candidates[0]}", flush=True)
        return str(candidates[0])
    print(f"no attached checkpoint; starting from {FALLBACK_MODEL}", flush=True)
    return FALLBACK_MODEL


def find_data_root() -> Path:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        available = [str(p) for p in Path("/kaggle/input").glob("*")]
        raise SystemExit(f"cannot find {ANNOTATION_NAME}. /kaggle/input holds: {available}")
    root = matches[0].parent.parent
    print(f"data root: {root}", flush=True)
    return root


def select_devices() -> list[int] | int:
    """Use every GPU the session was given."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no GPU attached; request a T4")

    major, minor = torch.cuda.get_device_capability()
    capability = f"sm_{major * 10 + minor}"
    if capability not in torch.cuda.get_arch_list():
        raise SystemExit(
            f"torch has no kernels for {capability} ({torch.cuda.get_device_name(0)}); "
            "set machine_shape to NvidiaTeslaT4"
        )

    count = torch.cuda.device_count()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {torch.cuda.get_device_name(0)} ({capability}) "
          f"x{count}, {total:.1f} GB each", flush=True)
    return list(range(count)) if count > 1 else 0


def main() -> None:
    data_root = find_data_root()
    devices = select_devices()

    for directory in (CKPT_DIR, OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])

    os.environ["YOLO_CONFIG_DIR"] = str(SCRATCH / "ultralytics")
    os.environ["PYTHONPATH"] = str(REPO_DIR)

    run([
        sys.executable, "-m", "experiments.exp_002_yolo_seg.src.prepare_yolo",
        "--annotations", data_root / "train" / ANNOTATION_NAME,
        "--images", data_root / "train" / "train_images",
        "--output", DATASET_DIR,
    ])

    from ultralytics import YOLO

    start_weights = find_start_weights()
    n_devices = len(devices) if isinstance(devices, list) else 1
    ladder = [b for b in BATCH_LADDER if b % n_devices == 0] or [n_devices]

    # Each attempt is charged against one shared budget. Out-of-memory normally
    # shows up in the first iterations, but if an attempt dies hours in, the next
    # rung must not restart with a fresh 10.75 hours and overrun the session cap.
    started = time.monotonic()

    for position, batch in enumerate(ladder):
        remaining = TIME_BUDGET_HOURS - (time.monotonic() - started) / 3600
        if remaining < 0.5:
            raise SystemExit(f"only {remaining:.2f} h left in the budget; not starting another attempt")
        try:
            print(f"\n=== training attempt {position + 1}/{len(ladder)}: batch={batch} "
                  f"at imgsz {IMGSZ}, {remaining:.2f} h remaining ===", flush=True)
            train_once(YOLO(start_weights), batch, devices, remaining)
            break
        except (RuntimeError, subprocess.CalledProcessError, torch_oom()) as error:
            if not is_out_of_memory(error) or position == len(ladder) - 1:
                raise
            print(f"out of memory at batch={batch}; falling back", flush=True)
            free_gpu()

    finish(data_root)


def torch_oom():
    import torch

    return getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def is_out_of_memory(error: Exception) -> bool:
    """Decide whether an exception is a memory failure worth retrying smaller.

    Under DDP, Ultralytics runs training in child processes via
    torch.distributed.run. A child that dies of CUDA OOM does not propagate its
    exception: the parent sees only

        subprocess.CalledProcessError: Command '[... torch.distributed.run ...]'
        returned non-zero exit status 1

    with the actual `torch.OutOfMemoryError` visible in the child's stderr and
    nowhere in the exception object. The first version of this check looked only
    for an OOM message on the exception and so never fired, which is why an
    out-of-memory at batch=8 killed the run instead of stepping down to 4.

    A CalledProcessError from the DDP launcher is therefore treated as a memory
    failure. That is a deliberate over-approximation: a genuine bug in the child
    would also be retried, but it would then fail identically at every rung and
    surface on the last one, costing a few minutes rather than a session.
    """
    if isinstance(error, subprocess.CalledProcessError):
        return "torch.distributed.run" in " ".join(map(str, error.cmd or []))
    return "out of memory" in str(error).lower()


def free_gpu() -> None:
    """Release cached blocks so the next rung starts from a clean allocator."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def train_once(model, batch: int, devices, budget_hours: float) -> None:
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=batch,
        device=devices,
        time=budget_hours,
        project=str(RUNS_DIR),
        name="exp003",
        exist_ok=True,
        cache=False,          # 974 images at 2048 will not fit in RAM
        workers=2,
        # Grayscale full-disk images: no hue or saturation to vary, and the
        # annotation convention has no up direction, so both flips are physical.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5, flipud=0.5, degrees=15.0,
        # Mosaic pastes four full-disk images into one frame, inventing limbs and
        # destroying the radial context that limb-darkening correction relies on.
        mosaic=0.0,
        patience=50,
        lr0=0.0005,
        warmup_epochs=1.0,
        seed=2026,
        verbose=True,
    )


def finish(data_root: Path) -> None:
    best = RUNS_DIR / "exp003" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training produced no checkpoint at {best}")
    shutil.copy2(best, CKPT_DIR / "best.pt")
    print(f"checkpoint: {(CKPT_DIR / 'best.pt').stat().st_size / 1e6:.1f} MB", flush=True)

    for name in ("results.csv", "args.yaml"):
        source = RUNS_DIR / "exp003" / name
        if source.exists():
            shutil.copy2(source, OUT_DIR / name)

    run([
        sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
        "--weights", CKPT_DIR / "best.pt",
        "--images", data_root / "test" / "test_images",
        "--output", OUT_DIR / "submission.csv",
        "--imgsz", IMGSZ,
        "--conf", "0.25",
        "--min-area", "150",
    ])
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
