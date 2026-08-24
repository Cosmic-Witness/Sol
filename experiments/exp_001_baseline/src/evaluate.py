"""Panoptic Quality evaluation against the held-out validation fold.

Two entry points:

`evaluate_pq`
    Called from the training loop each epoch on a fixed subset, to select the
    best checkpoint.

`main`
    Called from the command line after training, over the whole validation fold,
    to produce the number quoted in RESULTS.md, plus the per-image and
    per-instance distributions the final rubric asks for in section 1.6.

Scoring happens at the native 2048 frame, never at training resolution. A score
computed on 512-pixel masks would flatter every thin barb and would not
correspond to the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_001_baseline.src.postprocess import (  # noqa: E402
    annotations_to_instances,
    probability_to_instances,
)
from shared.utils import aggregate_pq, compute_pq  # noqa: E402


@lru_cache(maxsize=2)
def _load_coco(annotation_path: str) -> COCO:
    """COCO parsing takes several seconds. Cache it across epochs."""
    return COCO(annotation_path)


def _load_cached_image(cache_dir: str, stem: str) -> np.ndarray:
    path = Path(cache_dir) / "images" / f"{stem}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cache miss: {path}")
    return image


def _normalise(image: np.ndarray) -> np.ndarray:
    """Match the validation transform in dataset.py: ImageNet statistics, 3 channels."""
    x = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return x.transpose(2, 0, 1)


@torch.no_grad()
def evaluate_pq(
    model,
    cfg: dict,
    val_image_ids: list[str],
    id_to_stem: dict[str, str],
    device,
    subset: int | None = None,
    batch_size: int = 8,
    return_per_image: bool = False,
):
    """Aggregate PQ over validation records.

    `subset` takes the first N records of the fold. The fold order is
    deterministic, so the same records are scored every epoch and the
    epoch-to-epoch comparison stays fair.
    """
    coco = _load_coco(cfg["paths"]["annotations"])
    post = cfg["postprocess"]
    image_ids = val_image_ids[:subset] if subset else list(val_image_ids)

    model.eval()
    per_image: list[dict] = []

    for start in range(0, len(image_ids), batch_size):
        chunk = image_ids[start : start + batch_size]
        batch = np.stack(
            [_normalise(_load_cached_image(cfg["paths"]["cache_dir"], Path(id_to_stem[i]).stem)) for i in chunk]
        )
        tensor = torch.from_numpy(batch).to(device)
        with torch.autocast("cuda", enabled=tensor.is_cuda):
            logits = model(tensor)
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

        for image_id, probability in zip(chunk, probabilities):
            predicted = probability_to_instances(
                probability,
                threshold=post["threshold"],
                min_area=post["min_area"],
                closing_kernel=post["closing_kernel"],
                dilate_iterations=post["dilate_iterations"],
            )
            truth = annotations_to_instances(coco, image_id)
            result = compute_pq(predicted, truth)
            result["image_id"] = image_id
            result["n_pred"] = len(predicted)
            result["n_gt"] = len(truth)
            per_image.append(result)

    aggregate = aggregate_pq(per_image)
    if return_per_image:
        return aggregate, per_image
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a checkpoint on the validation fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset", type=int, default=None, help="limit to the first N records")
    args = parser.parse_args()

    from experiments.exp_001_baseline.src.model import build_model
    from experiments.exp_001_baseline.src.train import load_config
    from shared.data_split import load_records, make_split

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded epoch {state['epoch']}, recorded best PQ {state['best_pq']:.4f}")

    split = make_split(cfg["paths"]["annotations"], cfg["data"]["val_fraction"], cfg["seed"])
    _, id_to_stem = load_records(cfg["paths"]["annotations"])

    aggregate, per_image = evaluate_pq(
        model, cfg, split.val_image_ids, id_to_stem, device,
        subset=args.subset, return_per_image=True,
    )

    print(
        f"\nPQ {aggregate['pq']:.4f} | SQ {aggregate['sq']:.4f} | RQ {aggregate['rq']:.4f} | "
        f"TP {aggregate['tp']} FP {aggregate['fp']} FN {aggregate['fn']}"
    )

    # Distributions requested by the final rubric, section 1.6.
    pqs = np.array([r["pq"] for r in per_image])
    print(
        "per-image PQ: "
        + " ".join(f"p{p}={np.percentile(pqs, p):.3f}" for p in (10, 25, 50, 75, 90))
    )
    predicted_total = sum(r["n_pred"] for r in per_image)
    truth_total = sum(r["n_gt"] for r in per_image)
    print(f"instances predicted {predicted_total} vs ground truth {truth_total}")
    if truth_total:
        print(f"fragmentation ratio {predicted_total / truth_total:.2f} (1.00 is balanced)")

    out_path = Path(cfg["paths"]["log_dir"]) / "validation_per_image.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(per_image, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
