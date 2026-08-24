"""Metric, encoding, and reproducibility helpers shared by all experiments.

The Panoptic Quality implementation here is the single source of truth for local
scoring. Every experiment reports numbers produced by `compute_pq` so that results
across experiments stay comparable.
"""

from __future__ import annotations

import os
import random
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util

# The competition fixes every mask to the native GONG frame size.
IMAGE_SIZE = (2048, 2048)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG that can influence a training run.

    Torch is imported lazily so that submission-side tooling (which only needs
    RLE helpers) does not pay the cost of loading torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs throughput. We accept that cost: an experiment
    # whose result cannot be reproduced cannot be trusted as evidence.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# RLE encoding / decoding
# --------------------------------------------------------------------------- #
def encode_rle(binary_mask: np.ndarray) -> str:
    """Encode a binary mask as a COCO RLE `counts` string.

    Parameters
    ----------
    binary_mask
        Array of shape (H, W). Any non-zero value counts as foreground.

    Returns
    -------
    The `counts` field only. The submission format fixes `size` at 2048x2048,
    so the size is never written to the CSV.
    """
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    return mask_util.encode(mask)["counts"].decode("utf-8")


def decode_rle(counts: str, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Inverse of `encode_rle`. Returns a uint8 mask of shape `size`."""
    rle = {"size": list(size), "counts": counts.encode("utf-8")}
    return mask_util.decode(rle)


def _to_rle_objects(masks: Sequence[np.ndarray]) -> list[dict]:
    """Encode a list of binary masks into pycocotools RLE dicts."""
    if not masks:
        return []
    stacked = np.stack([m.astype(np.uint8) for m in masks], axis=-1)
    return mask_util.encode(np.asfortranarray(stacked))


# --------------------------------------------------------------------------- #
# Panoptic Quality
# --------------------------------------------------------------------------- #
def compute_pq(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict:
    """Panoptic Quality for one image, following Kirillov et al. (CVPR 2019).

        PQ = sum(IoU over matched pairs) / (|TP| + 0.5*|FP| + 0.5*|FN|)

    Matching note: above an IoU of 0.5 a prediction can exceed the threshold with
    at most one ground-truth segment, and vice versa. The match is therefore
    unique, and greedy selection by descending IoU is provably optimal. No
    Hungarian solver is needed.

    Both inputs are lists of binary arrays. Empty predictions score PQ 0 whenever
    ground truth exists, which is the behaviour the competition intends.

    Returns
    -------
    dict with keys pq, sq, rq, tp, fp, fn, iou_sum.
    """
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return {
            "pq": 0.0,
            "sq": 0.0,
            "rq": 0.0,
            "tp": 0,
            "fp": n_pred,
            "fn": n_gt,
            "iou_sum": 0.0,
        }

    # pycocotools computes the full IoU matrix in C, which is far cheaper than
    # a Python double loop over 2048x2048 arrays.
    ious = np.asarray(
        mask_util.iou(
            _to_rle_objects(list(pred_masks)),
            _to_rle_objects(list(gt_masks)),
            [0] * n_gt,
        )
    ).reshape(n_pred, n_gt)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    iou_sum = 0.0

    # Consider only candidate pairs above threshold, best first.
    pairs = np.argwhere(ious > iou_threshold)
    for i, j in sorted(pairs, key=lambda p: -ious[p[0], p[1]]):
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(int(i))
        matched_gt.add(int(j))
        iou_sum += float(ious[i, j])

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - tp
    sq = iou_sum / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return {
        "pq": sq * rq,
        "sq": sq,
        "rq": rq,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_sum": iou_sum,
    }


def aggregate_pq(per_image: Iterable[dict]) -> dict:
    """Combine per-image PQ components into a dataset-level score.

    The leaderboard pools counts across the whole test set rather than averaging
    per-image PQ values. Pooling matters: an image with one filament would
    otherwise carry the same weight as an image with twenty.
    """
    tp = fp = fn = 0
    iou_sum = 0.0
    for r in per_image:
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
        iou_sum += r["iou_sum"]
    sq = iou_sum / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- #
# Overlap resolution and validation
# --------------------------------------------------------------------------- #
def paint_panoptic(
    candidates: Sequence[tuple[float, np.ndarray]],
    min_area: int = 150,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> list[tuple[float, np.ndarray, int]]:
    """Resolve overlapping instance masks into a pixel-disjoint set.

    Candidates are painted onto a shared canvas in descending score order. Each
    mask keeps only the pixels no higher-scoring mask already claimed. The
    scorer rejects overlapping predictions, so this step is mandatory for any
    model that can emit overlaps.

    `min_area` default of 150 px is set from the training distribution: the 1st
    percentile ground-truth instance covers 208 px, so 150 discards fragments
    without discarding real small filaments.
    """
    ordered = sorted(candidates, key=lambda c: -c[0])
    claimed = np.zeros(image_size, dtype=np.uint8)
    kept: list[tuple[float, np.ndarray, int]] = []
    for score, binary in ordered:
        remaining = (binary.astype(np.uint8) & (1 - claimed)).astype(np.uint8)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append((score, remaining, area))
    return kept


def write_submission(predictions: dict[str, Sequence[np.ndarray]], csv_path: str) -> pd.DataFrame:
    """Write the competition CSV.

    `predictions` maps an image stem (no extension) to its list of disjoint
    binary masks. Filament ids are `<stem>_<N>`, numbered from 1.
    """
    rows = []
    for stem, masks in predictions.items():
        for idx, binary in enumerate(masks, start=1):
            rows.append(
                {"filament_id": f"{stem}_{idx}", "segmentation_rle": encode_rle(binary)}
            )
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(csv_path, index=False)
    return df


def check_no_overlap(csv_path: str, image_size: tuple[int, int] = IMAGE_SIZE) -> None:
    """Fail loudly if any image in the CSV contains overlapping masks.

    Run this before every submission. A rejected submission costs a slot, and
    slots are the scarcest resource in the competition.
    """
    df = pd.read_csv(csv_path)
    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    for stem, group in df.groupby("stem"):
        if len(group) < 2:
            continue
        acc = np.zeros(image_size, dtype=np.int16)
        for counts in group["segmentation_rle"]:
            acc += decode_rle(counts, image_size).astype(np.int16)
        if (acc > 1).any():
            raise ValueError(f"Overlapping masks in image {stem}")
    print(f"PASS: no overlaps across {df['stem'].nunique()} images, {len(df)} instances")
