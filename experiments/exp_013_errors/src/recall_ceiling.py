"""How much of the miss is the detector not seeing, and how much is it not believing?

285 ground-truth filaments are overlapped by no prediction at all. But
"prediction" there means a candidate that survived confidence 0.35 and the
300-pixel area floor. The cached candidate pool runs down to confidence 0.05, so
the question splits in two: does the detector propose something at those places
and rank it too low, or does it propose nothing at all?

The answer decides whether a better ranker is worth building. A pool that already
covers the misses puts a ceiling on re-ranking that is worth chasing; a pool that
does not means the misses are invisible to this detector and only a different
detector reaches them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util

from shared.data_split import make_split
from shared.utils import decode_rle

FULL = 2048
SHIPPED_CONF = 0.35
MATCH = 0.5
TOUCH = 0.10


def gt_masks(annotations: list[dict]) -> list[np.ndarray]:
    out = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not segmentation or isinstance(segmentation, dict):
            continue
        rings = [r for r in segmentation if len(r) >= 6]
        if not rings:
            continue
        rles = mask_util.frPyObjects(rings, FULL, FULL)
        out.append(mask_util.decode(mask_util.merge(rles)).astype(np.uint8))
    return out


def iou_matrix(left: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)))
    a = mask_util.encode(np.asfortranarray(np.stack(left, axis=-1)))
    b = mask_util.encode(np.asfortranarray(np.stack(right, axis=-1)))
    return mask_util.iou(a, b, [0] * len(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}
    split = make_split(args.annotations)

    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    # Candidates are cached raw, before painting: overlapping and unfiltered.
    # That is what a re-ranker would choose from, so it is what bounds it.
    pool: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    for stem, entries in cached.items():
        masks = [decode_rle(e["counts"]) for e in entries]
        scores = np.array([e["features"]["score"] for e in entries])
        pool[stem] = (masks, scores)

    counts = {
        "truths": 0,
        "covered_by_pool_at_match": 0,      # some candidate anywhere clears 0.5
        "covered_by_pool_at_touch": 0,      # some candidate anywhere overlaps
        "covered_above_shipped_conf": 0,    # ... and it survives conf 0.35
        "invisible": 0,                     # nothing in the pool touches it
    }
    best_rank_of_covered: list[int] = []
    scores_of_covering: list[float] = []
    pool_sizes: list[int] = []

    for image_id in split.val_image_ids:
        truths = gt_masks(by_image.get(image_id, []))
        masks, scores = pool.get(stem_of[image_id], ([], np.zeros(0)))
        counts["truths"] += len(truths)
        pool_sizes.append(len(masks))
        if not truths:
            continue
        if not masks:
            counts["invisible"] += len(truths)
            continue
        matrix = iou_matrix(masks, truths)          # candidates x truths
        order = np.argsort(-scores)
        rank_of = np.empty(len(scores), dtype=int)
        rank_of[order] = np.arange(len(scores))

        for j in range(len(truths)):
            column = matrix[:, j]
            best = int(column.argmax())
            if column[best] >= MATCH:
                counts["covered_by_pool_at_match"] += 1
                best_rank_of_covered.append(int(rank_of[best]))
                scores_of_covering.append(float(scores[best]))
                if scores[best] >= SHIPPED_CONF:
                    counts["covered_above_shipped_conf"] += 1
            elif column[best] >= TOUCH:
                counts["covered_by_pool_at_touch"] += 1
            else:
                counts["invisible"] += 1

    total = counts["truths"] or 1
    report = {
        "truths": counts["truths"],
        "mean_candidates_per_photograph": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "pool_covers_at_iou_0.5": {
            "count": counts["covered_by_pool_at_match"],
            "share": counts["covered_by_pool_at_match"] / total},
        "of_those_already_above_conf_0.35": {
            "count": counts["covered_above_shipped_conf"],
            "share": counts["covered_above_shipped_conf"] / max(counts["covered_by_pool_at_match"], 1)},
        "pool_only_grazes": {
            "count": counts["covered_by_pool_at_touch"],
            "share": counts["covered_by_pool_at_touch"] / total},
        "invisible_to_the_detector": {
            "count": counts["invisible"], "share": counts["invisible"] / total},
        "covering_candidate_confidence": {
            "median": float(np.median(scores_of_covering)) if scores_of_covering else None,
            "p10": float(np.percentile(scores_of_covering, 10)) if scores_of_covering else None,
        },
        # A perfect re-ranker would keep exactly the covering candidate for each
        # truth and nothing else. Its RQ is the coverage share; the recall it
        # cannot exceed is the same number.
        "oracle_reranker_recall": counts["covered_by_pool_at_match"] / total,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
