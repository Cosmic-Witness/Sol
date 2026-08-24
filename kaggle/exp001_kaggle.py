"""Kaggle kernel driver for experiment 001.

This script is the whole kernel. It is deliberately thin: every algorithm lives
in the repository it clones, so the code that produced a leaderboard score is
the code in git, not a copy pasted into a notebook cell.

Kaggle specifics this driver handles
------------------------------------
- Competition data mounts read-only at /kaggle/input/<competition-slug>/.
- Only /kaggle/working persists, and it becomes the kernel output.
- A run is capped at 12 hours. Training checkpoints every epoch so a second run
  can continue where the first stopped.
- Resume works by attaching this kernel's own previous output as a source. The
  driver looks for checkpoints under /kaggle/input/ and copies them in before
  training starts.

Run locally for a syntax check only. It expects the Kaggle filesystem.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Cosmic-Witness/Sol"
BRANCH = "exp-001-baseline"
REPO_DIR = Path("/kaggle/working/Sol")

WORKING = Path("/kaggle/working")
CACHE_DIR = WORKING / "cache"
CKPT_DIR = WORKING / "checkpoints"
LOG_DIR = WORKING / "logs"
OUT_DIR = WORKING / "outputs"

COMPETITION = "filament-segmentation-2026"
ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def find_data_root() -> Path:
    """Locate the competition data by searching for its annotation file.

    Kaggle does not mount a competition where the obvious guess puts it. The
    first run of this kernel assumed /kaggle/input/<slug>/ and died on a
    FileNotFoundError; a probe kernel showed the real location is
    /kaggle/input/competitions/<slug>/. Searching for a known filename removes
    the guess entirely and survives any future change to the mount layout.
    """
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        available = [str(p) for p in Path("/kaggle/input").glob("*")]
        raise SystemExit(
            f"cannot find {ANNOTATION_NAME} under /kaggle/input. "
            f"Is the competition attached? Top level holds: {available}"
        )
    # <root>/train/<annotation file>  ->  <root>
    root = matches[0].parent.parent
    print(f"data root discovered: {root}", flush=True)
    return root


DATA_ROOT = Path("/kaggle/input")  # replaced in main() by find_data_root()


def run(command: list[str]) -> None:
    """Run a subprocess and fail the kernel loudly if it errors.

    Kaggle marks a kernel complete even when a step failed quietly, so an
    explicit check keeps a broken run from looking like a finished one.
    """
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=str(REPO_DIR) if REPO_DIR.exists() else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}: {' '.join(command)}")


def report_environment() -> None:
    print("=" * 70, flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    import torch

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        # Training on CPU here would burn the 12-hour budget for nothing.
        raise SystemExit("no GPU attached; set the accelerator to T4 and re-run")
    print(f"data root exists: {DATA_ROOT.exists()}", flush=True)
    print("=" * 70, flush=True)


def restore_previous_run() -> None:
    """Copy checkpoints, logs, and cache from an attached previous kernel output.

    Kaggle mounts every attached source under /kaggle/input/. A previous run of
    this kernel appears there as a directory holding the working tree it left
    behind. Copying it forward is what makes a multi-run training possible
    inside the 12-hour cap.
    """
    for candidate in sorted(Path("/kaggle/input").glob("*")):
        # "competitions" holds the read-only competition mount, never a prior run.
        if candidate.name in (COMPETITION, "competitions"):
            continue
        for name, destination in (
            ("checkpoints", CKPT_DIR),
            ("cache", CACHE_DIR),
            ("logs", LOG_DIR),
        ):
            source = candidate / name
            if source.is_dir() and not destination.exists():
                print(f"restoring {name} from {candidate.name}", flush=True)
                shutil.copytree(source, destination)


def write_config() -> Path:
    """Derive a Kaggle config from the committed one, changing only paths."""
    import yaml

    base = REPO_DIR / "experiments/exp_001_baseline/config.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["paths"].update(
        {
            "data_root": str(DATA_ROOT),
            "annotations": str(DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
            "cache_dir": str(CACHE_DIR),
            "checkpoint_dir": str(CKPT_DIR),
            "log_dir": str(LOG_DIR),
            "output_dir": str(OUT_DIR),
        }
    )
    # Kaggle allocates 4 vCPUs alongside the T4. Two workers keeps the loader
    # ahead of the GPU without starving the main process.
    cfg["data"]["num_workers"] = 2

    target = REPO_DIR / "experiments/exp_001_baseline/config_kaggle.yaml"
    target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"\nconfig written to {target}", flush=True)
    return target


def main() -> None:
    global DATA_ROOT
    DATA_ROOT = find_data_root()
    report_environment()

    for directory in (CKPT_DIR, LOG_DIR, OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    restore_previous_run()

    run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "albumentations"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)
    run(["git", "log", "--oneline", "-1"])

    config = write_config()

    run([sys.executable, "-m", "experiments.exp_001_baseline.src.prepare_cache", "--config", str(config)])
    run([sys.executable, "-m", "experiments.exp_001_baseline.src.train", "--config", str(config)])

    best = CKPT_DIR / "best.pt"
    if not best.exists():
        raise SystemExit("training produced no best checkpoint")

    run([sys.executable, "-m", "experiments.exp_001_baseline.src.evaluate",
         "--config", str(config), "--checkpoint", str(best)])
    run([sys.executable, "-m", "experiments.exp_001_baseline.src.predict",
         "--config", str(config), "--checkpoint", str(best),
         "--images", str(DATA_ROOT / "test/test_images"),
         "--output", str(OUT_DIR / "submission.csv"), "--tta"])

    # The clone would otherwise be copied into the kernel output, which wastes
    # space and confuses the next run's restore step.
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    print("\nDONE. submission at", OUT_DIR / "submission.csv", flush=True)


if __name__ == "__main__":
    main()
