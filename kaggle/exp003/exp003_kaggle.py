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

MODEL = "yolo11l-seg.pt"
IMGSZ = 2048
EPOCHS = 300          # never reached; the time budget is the real stop condition
BATCH = -1            # autobatch: largest that fits, measured not guessed
TIME_BUDGET_HOURS = 10.75


def run(command: list, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


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

    model = YOLO(MODEL)
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=devices,
        time=TIME_BUDGET_HOURS,
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
        seed=2026,
        verbose=True,
    )

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
