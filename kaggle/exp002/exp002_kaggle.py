"""Kaggle kernel driver for experiment 002: YOLO instance segmentation.

Like the exp_001 driver this is deliberately thin. Every algorithm lives in the
repository it clones, so the code behind a leaderboard score is the code in git.

Why this experiment exists
--------------------------
exp_001 predicts a binary map and recovers instances with connected components.
Feeding perfect ground truth through that path reaches PQ 0.76 at 512 and 0.87 at
1024 — a hard ceiling imposed by the design, not by the model's accuracy, because
connectivity cannot rejoin a filament the sky broke into pieces. Predicting
instances directly removes the ceiling. The strongest public solution on this
leaderboard is a YOLO segmentation model, which is corroborating evidence that
the family is right for the task.

Kaggle specifics handled here
-----------------------------
- Competition data mounts read-only; its exact path is discovered, not assumed.
- Only /kaggle/working persists and becomes the kernel output, so the dataset and
  the clone live on /kaggle/temp scratch and never bloat the download.
- Training is capped by wall clock, below the session limit, so the run always
  reaches export and prediction instead of being killed mid-epoch with nothing
  to show.
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

# Training shape. 1280 is a compromise: barbs are a few pixels wide at 2048, so
# resolution buys recall, but a T4 holds only a small batch at 1536 and the
# epoch count matters more than the last few pixels of detail.
MODEL = "yolo11m-seg.pt"
IMGSZ = 1280
EPOCHS = 100
BATCH = 4
# Below Kaggle's 12-hour cap with room for export, prediction over 180 test
# images, and the output upload.
TIME_BUDGET_HOURS = 8.5


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess and fail the kernel loudly if it errors."""
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def find_data_root() -> Path:
    """Locate the competition data by searching for its annotation file.

    Kaggle does not mount a competition where the obvious guess puts it; the real
    location is /kaggle/input/competitions/<slug>/<dataset dir>/. Searching for a
    known filename removes the guess and survives any change to the layout.
    """
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        available = [str(p) for p in Path("/kaggle/input").glob("*")]
        raise SystemExit(f"cannot find {ANNOTATION_NAME}. /kaggle/input holds: {available}")
    root = matches[0].parent.parent
    print(f"data root: {root}", flush=True)
    return root


def report_environment() -> None:
    print("=" * 70, flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    import torch

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("no GPU attached; request a T4 and re-run")

    # A P100 reports cuda available and then dies at the first convolution,
    # because sm_60 is below the floor of the cu128 build in the Kaggle image.
    # exp_001 lost 24 minutes to that before the check existed.
    major, minor = torch.cuda.get_device_capability()
    capability = f"sm_{major * 10 + minor}"
    architectures = torch.cuda.get_arch_list()
    print(f"device: {torch.cuda.get_device_name(0)} ({capability})", flush=True)
    if capability not in architectures:
        raise SystemExit(
            f"torch has no kernels for {capability}; it supports {architectures}. "
            "Set machine_shape to NvidiaTeslaT4 in kernel-metadata.json."
        )
    print("=" * 70, flush=True)


def main() -> None:
    data_root = find_data_root()
    report_environment()

    for directory in (CKPT_DIR, OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])

    # Ultralytics writes settings and datasets under its config dir; both must be
    # writable, and neither belongs in the kernel output.
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
        time=TIME_BUDGET_HOURS,
        project=str(RUNS_DIR),
        name="exp002",
        exist_ok=True,
        # Grayscale solar disks: colour jitter models a variation that does not
        # exist, and vertical flips are physically fine because the annotation
        # convention has no up direction.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5, flipud=0.5, degrees=15.0,
        # Mosaic pastes four images into one frame. On a full-disk image that
        # invents limbs and destroys the radial context the model needs, so it is
        # off for the whole run rather than only at the end.
        mosaic=0.0,
        patience=25,
        seed=2026,
        verbose=True,
    )

    best = RUNS_DIR / "exp002" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training produced no checkpoint at {best}")
    shutil.copy2(best, CKPT_DIR / "best.pt")
    print(f"checkpoint saved: {(CKPT_DIR / 'best.pt').stat().st_size / 1e6:.1f} MB", flush=True)

    # Keep the training curves; they are small and they are the record of the run.
    for name in ("results.csv", "args.yaml"):
        source = RUNS_DIR / "exp002" / name
        if source.exists():
            shutil.copy2(source, OUT_DIR / name)

    run([
        sys.executable, "-m", "experiments.exp_002_yolo_seg.src.predict",
        "--weights", CKPT_DIR / "best.pt",
        "--images", data_root / "test" / "test_images",
        "--output", OUT_DIR / "submission.csv",
        "--imgsz", IMGSZ,
        "--conf", "0.25",
        "--iou", "0.60",
        "--min-area", "150",
    ])

    print("\nDONE. submission at", OUT_DIR / "submission.csv", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # The clone and the dataset are on scratch, but be explicit: a stray copy
        # in the output makes `kaggle kernels output` slow enough to block
        # diagnosis, which is what happened on the first exp_001 run.
        shutil.rmtree(REPO_DIR, ignore_errors=True)
