"""Build the conditioned training cache. Run once before training.

Conditioning a 2048x2048 frame costs about a second, and the cache is read
roughly 60 times per observation over a full run. Doing the work once turns a
preprocessing-bound job into a GPU-bound one.

The step is resumable: files already present are skipped. On Colab, point
`cache_dir` at Google Drive so a reclaimed runtime does not discard the work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_001_baseline.src.dataset import build_cache  # noqa: E402
from experiments.exp_001_baseline.src.train import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the conditioned image cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--images",
        default=None,
        help="training image directory; defaults to <data_root>/train/train_images",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_dir = args.images or str(Path(cfg["paths"]["data_root"]) / "train" / "train_images")

    build_cache(
        annotation_path=cfg["paths"]["annotations"],
        images_dir=images_dir,
        cache_dir=cfg["paths"]["cache_dir"],
        image_size=cfg["data"]["image_size"],
        clahe_clip=cfg["preprocess"]["clahe_clip"],
        clahe_grid=cfg["preprocess"]["clahe_grid"],
        blur_sigma=cfg["preprocess"]["blur_sigma"],
    )
    cache = Path(cfg["paths"]["cache_dir"])
    print(
        f"cache ready: {len(list((cache / 'images').glob('*.png')))} images, "
        f"{len(list((cache / 'masks').glob('*.png')))} masks"
    )


if __name__ == "__main__":
    main()
