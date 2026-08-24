"""CPU-only kernel that builds the conditioning cache and publishes it as a zip.

Why this is a separate kernel
-----------------------------
Conditioning 707 observations is CPU work. Running it inside the GPU kernel
spent roughly twenty minutes of accelerator quota per attempt on a step that
never touches the GPU, and it put thousands of small PNGs into the kernel
output, which made `kaggle kernels output` too slow to retrieve the run log.

This kernel does that work with no accelerator, writes the cache to scratch, and
publishes a single archive. The training kernel attaches this one's output and
starts at epoch 0 immediately.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Cosmic-Witness/Sol"
BRANCH = "exp-001-baseline"
REPO_DIR = Path("/kaggle/working/Sol")
CACHE_DIR = Path("/kaggle/temp/cache")
ARCHIVE = Path("/kaggle/working/cache_512.zip")
ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"step failed with code {result.returncode}")


def find_data_root() -> Path:
    matches = sorted(Path("/kaggle/input").rglob(ANNOTATION_NAME))
    if not matches:
        raise SystemExit(f"{ANNOTATION_NAME} not found under /kaggle/input")
    root = matches[0].parent.parent
    print(f"data root: {root}", flush=True)
    return root


def main() -> None:
    data_root = find_data_root()
    images_dir = data_root / "train" / "train_images"
    print(f"train images: {sum(1 for _ in images_dir.iterdir())}", flush=True)

    run([sys.executable, "-m", "pip", "install", "-q",
         "segmentation-models-pytorch", "albumentations"])
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])

    import yaml

    base = REPO_DIR / "experiments/exp_001_baseline/config.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["paths"]["annotations"] = str(data_root / "train" / ANNOTATION_NAME)
    cfg["paths"]["cache_dir"] = str(CACHE_DIR)
    config = REPO_DIR / "experiments/exp_001_baseline/config_cache.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    run([sys.executable, "-m", "experiments.exp_001_baseline.src.prepare_cache",
         "--config", str(config), "--images", str(images_dir)], cwd=REPO_DIR)

    print("\narchiving", flush=True)
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", str(CACHE_DIR))
    print(f"{ARCHIVE.name}: {ARCHIVE.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Keep the output to the archive alone, so it downloads in seconds and
        # the run log is always reachable, including after a failure.
        shutil.rmtree(REPO_DIR, ignore_errors=True)
