"""Measure the ceiling of the semantic-then-label design, before training it.

The question this answers
------------------------
Experiment 001 predicts a binary mask at reduced resolution, then labels
connected components. Two steps lose information before the model is even
involved:

1. Downsampling to `image_size` blurs every filament boundary, which caps IoU
   and therefore caps SQ.
2. Connected-component labelling cannot separate two filaments that touch, and
   cannot join one filament that imaging noise broke apart, which caps RQ.

Feeding the ground-truth mask through the same path measures both caps directly.
Whatever PQ comes out is the best score this design could reach with a perfect
segmentation model. Training cannot exceed it.

Run:

    python -m experiments.exp_001_baseline.src.ceiling_analysis \
        --config experiments/exp_001_baseline/config.yaml --records 60

Only the annotation file is needed. No images, no GPU, no trained weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_001_baseline.src.postprocess import (  # noqa: E402
    annotations_to_instances,
    probability_to_instances,
)
from shared.data_split import make_split  # noqa: E402
from shared.utils import aggregate_pq, compute_pq  # noqa: E402


def measure(coco: COCO, image_ids: list[str], image_size: int, closing_kernel: int, min_area: int) -> dict:
    """PQ obtainable when the semantic mask is exactly right."""
    per_image = []
    for image_id in image_ids:
        truth = annotations_to_instances(coco, image_id)
        union = np.zeros((2048, 2048), dtype=np.uint8)
        for mask in truth:
            union |= mask

        # Exactly the path dataset.py uses to build a training target.
        small = cv2.resize(union * 255, (image_size, image_size), interpolation=cv2.INTER_AREA)
        perfect_probability = (small > 32).astype(np.float32)

        predicted = probability_to_instances(
            perfect_probability, threshold=0.5, min_area=min_area, closing_kernel=closing_kernel
        )
        result = compute_pq(predicted, truth)
        result["n_pred"] = len(predicted)
        result["n_gt"] = len(truth)
        per_image.append(result)

    summary = aggregate_pq(per_image)
    summary["n_pred"] = sum(r["n_pred"] for r in per_image)
    summary["n_gt"] = sum(r["n_gt"] for r in per_image)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the design ceiling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", type=int, default=60, help="validation records to score")
    args = parser.parse_args()

    from experiments.exp_001_baseline.src.train import load_config

    cfg = load_config(args.config)
    coco = COCO(cfg["paths"]["annotations"])
    split = make_split(cfg["paths"]["annotations"], cfg["data"]["val_fraction"], cfg["seed"])
    image_ids = split.val_image_ids[: args.records]
    min_area = cfg["postprocess"]["min_area"]

    print(f"ceiling over {len(image_ids)} validation records, min_area {min_area}\n")
    header = f"{'size':>6} {'closing':>8} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'TP':>5} {'FP':>4} {'FN':>4} {'pred/gt':>8}"
    print(header)
    print("-" * len(header))
    for image_size in (512, 1024):
        for closing_kernel in (1, 5, 9):
            r = measure(coco, image_ids, image_size, closing_kernel, min_area)
            print(
                f"{image_size:>6} {closing_kernel:>8} {r['pq']:>7.4f} {r['sq']:>7.4f} {r['rq']:>7.4f} "
                f"{r['tp']:>5} {r['fp']:>4} {r['fn']:>4} {r['n_pred'] / max(r['n_gt'], 1):>8.2f}"
            )


if __name__ == "__main__":
    main()
