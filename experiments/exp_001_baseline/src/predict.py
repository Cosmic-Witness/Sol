"""Test-set inference and submission writing for experiment 001.

Run from the repository root:

    python -m experiments.exp_001_baseline.src.predict \
        --config experiments/exp_001_baseline/config.yaml \
        --checkpoint <path to best.pt> \
        --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
        --output experiments/exp_001_baseline/outputs/submission.csv

Test frames are conditioned here rather than read from the training cache. The
cache holds training observations only, and inference must touch nothing but the
images in the test directory, per the competition rules in section 1.7.

The overlap validator runs before the file is considered finished. A submission
with overlapping masks is rejected by the scorer, and a rejected submission
still consumes a daily slot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_001_baseline.src.model import build_model  # noqa: E402
from experiments.exp_001_baseline.src.postprocess import probability_to_instances  # noqa: E402
from experiments.exp_001_baseline.src.train import load_config  # noqa: E402
from shared.preprocessing import preprocess  # noqa: E402
from shared.utils import check_no_overlap, write_submission  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _to_tensor(image: np.ndarray) -> np.ndarray:
    x = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


@torch.no_grad()
def predict_probability(model, image: np.ndarray, device, tta: bool) -> np.ndarray:
    """Forward one conditioned image, optionally averaging flip augmentations.

    Test-time augmentation over the two flips and their combination costs four
    forward passes and needs no retraining. It usually lifts SQ slightly by
    smoothing boundary noise.
    """
    tensor = torch.from_numpy(_to_tensor(image))[None].to(device)
    variants = [(tensor, None)]
    if tta:
        variants += [
            (torch.flip(tensor, dims=[3]), [3]),
            (torch.flip(tensor, dims=[2]), [2]),
            (torch.flip(tensor, dims=[2, 3]), [2, 3]),
        ]

    accumulated = None
    for batch, flip_dims in variants:
        with torch.autocast("cuda", enabled=batch.is_cuda):
            logits = model(batch)
        probability = torch.sigmoid(logits.float())
        if flip_dims:
            probability = torch.flip(probability, dims=flip_dims)
        accumulated = probability if accumulated is None else accumulated + probability
    return (accumulated / len(variants)).cpu().numpy()[0, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a submission CSV from a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True, help="directory of test JPEGs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tta", action="store_true", help="average four flip variants")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre = cfg["preprocess"]
    post = cfg["postprocess"]
    size = cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded epoch {state['epoch']}, validation PQ {state['best_pq']:.4f}", flush=True)

    test_paths = sorted(Path(args.images).glob("*.jpeg")) + sorted(Path(args.images).glob("*.jpg"))
    if not test_paths:
        raise FileNotFoundError(f"no test images under {args.images}")
    print(f"{len(test_paths)} test images", flush=True)

    predictions: dict[str, list[np.ndarray]] = {}
    empty_images: list[str] = []

    for n, path in enumerate(test_paths, start=1):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(path)

        conditioned, disk = preprocess(
            raw, pre["clahe_clip"], pre["clahe_grid"], pre["blur_sigma"], return_disk=True
        )
        resized = cv2.resize(conditioned, (size, size), interpolation=cv2.INTER_AREA)
        probability = predict_probability(model, resized, device, args.tta)

        instances = probability_to_instances(
            probability,
            threshold=post["threshold"],
            min_area=post["min_area"],
            closing_kernel=post["closing_kernel"],
            dilate_iterations=post["dilate_iterations"],
            disk=disk,
            output_size=raw.shape[:2],
        )
        if not instances:
            empty_images.append(path.stem)
        predictions[path.stem] = instances

        if n % 20 == 0:
            print(f"predicted {n}/{len(test_paths)}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame = write_submission(predictions, args.output)

    counts = [len(v) for v in predictions.values()]
    print(
        f"\nwrote {args.output}: {len(frame)} instances across {len(predictions)} images "
        f"(mean {np.mean(counts):.1f}, max {max(counts)})"
    )
    if empty_images:
        # Every image without a prediction contributes only false negatives.
        # The training fold has no record with zero filaments, so an empty
        # prediction is a model failure rather than an empty sky.
        print(f"WARNING: {len(empty_images)} images received no prediction: {empty_images[:5]}")

    check_no_overlap(args.output)


if __name__ == "__main__":
    main()
