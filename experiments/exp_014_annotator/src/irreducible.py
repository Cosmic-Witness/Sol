"""Are the detector's phantom filaments phantoms, or are they another annotator's?

The error decomposition found that 49% of false positives overlap no ground
truth at all, and 59% of false negatives are overlapped by no prediction at all.
Neither class can be reached by any boundary method, which is where every
experiment so far has spent its compute.

But "overlaps no ground truth" is measured against one annotator, and 296 of the
707 photographs were labelled independently by two or three people who agree with
each other at PQ 0.337. A prediction that annotator A never drew may be exactly
what annotator B drew.

This measures that, with the control that matters: the same statistic computed
for one human against another. If people disagree at the detector's rate, the
class is irreducible and the compute belongs on mask quality. If people agree far
better than the detector does, the detector is genuinely missing and inventing
filaments, and that is where the score is.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util

from shared.data_split import make_split
from shared.utils import decode_rle, paint_panoptic

FULL = 2048
CONF = 0.35
MIN_AREA = 300
GROW = -1
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


def erode(binary: np.ndarray, iterations: int) -> np.ndarray:
    if iterations >= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(iterations) + 1,) * 2)
    return cv2.erode(binary, kernel).astype(np.uint8)


def iou_matrix(left: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)))
    a = mask_util.encode(np.asfortranarray(np.stack(left, axis=-1)))
    b = mask_util.encode(np.asfortranarray(np.stack(right, axis=-1)))
    return mask_util.iou(a, b, [0] * len(right))


def orphan_rate(source: list[np.ndarray], reference: list[np.ndarray]) -> tuple[int, int]:
    """How many of `source` overlap nothing in `reference` at all.

    Returns (orphans, total). An orphan is the strongest possible failure: not a
    misplaced boundary but an object the other side does not acknowledge exists.
    """
    if not source:
        return 0, 0
    if not reference:
        return len(source), len(source)
    best = iou_matrix(source, reference).max(axis=1)
    return int((best < TOUCH).sum()), len(source)


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
    ids_of_stem = defaultdict(list)
    for record in coco["images"]:
        ids_of_stem[record["file_name"]].append(record["id"])

    split = make_split(args.annotations)
    val_stems = set(split.val_stems)

    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    tally = {
        # detector against one annotator, on the same photographs the human
        # comparison uses, so the two rows are directly comparable
        "pred_orphans": 0, "pred_total": 0,
        "truth_orphans": 0, "truth_total": 0,
        # human against human, both directions
        "human_orphans": 0, "human_total": 0,
        # the reprieve: predictions orphaned by A that some other annotator drew
        "pred_orphans_rescued": 0,
        "truth_orphans_rescued": 0,
        "photographs": 0, "multi_photographs": 0,
    }

    for stem in sorted(val_stems):
        record_ids = ids_of_stem.get(stem, [])
        if not record_ids:
            continue
        candidates = []
        for entry in cached.get(stem, []):
            score = entry["features"]["score"]
            if score < CONF:
                continue
            candidates.append((score, erode(decode_rle(entry["counts"]), GROW)))
        preds = [m for _, m, _ in paint_panoptic(candidates, MIN_AREA)]
        truths = {rid: gt_masks(by_image.get(rid, [])) for rid in record_ids}
        tally["photographs"] += 1

        for rid in record_ids:
            o, t = orphan_rate(preds, truths[rid])
            tally["pred_orphans"] += o
            tally["pred_total"] += t
            o, t = orphan_rate(truths[rid], preds)
            tally["truth_orphans"] += o
            tally["truth_total"] += t

        if len(record_ids) < 2:
            continue
        tally["multi_photographs"] += 1

        # Human against human, every ordered pair.
        for a in record_ids:
            for b in record_ids:
                if a == b:
                    continue
                o, t = orphan_rate(truths[a], truths[b])
                tally["human_orphans"] += o
                tally["human_total"] += t

        # A prediction orphaned by every annotator is a genuine invention. One
        # that some annotator drew is a disagreement between people that the
        # detector happened to land on one side of.
        union = [m for rid in record_ids for m in truths[rid]]
        for rid in record_ids:
            if not preds:
                continue
            best_here = iou_matrix(preds, truths[rid]).max(axis=1) if truths[rid] else np.zeros(len(preds))
            best_any = iou_matrix(preds, union).max(axis=1) if union else np.zeros(len(preds))
            tally["pred_orphans_rescued"] += int(((best_here < TOUCH) & (best_any >= TOUCH)).sum())

            others = [m for other in record_ids if other != rid for m in truths[other]]
            if truths[rid]:
                mine_vs_pred = iou_matrix(truths[rid], preds).max(axis=1)
                mine_vs_others = (iou_matrix(truths[rid], others).max(axis=1)
                                  if others else np.zeros(len(truths[rid])))
                # A truth the detector missed that no other annotator drew either
                # is a label only one person believed in.
                tally["truth_orphans_rescued"] += int(
                    ((mine_vs_pred < TOUCH) & (mine_vs_others < TOUCH)).sum())

    def rate(orphans: int, total: int) -> float:
        return orphans / total if total else 0.0

    report = {
        "photographs": tally["photographs"],
        "multiply_annotated": tally["multi_photographs"],
        "detector_predictions_orphaned_by_the_annotator": {
            "count": tally["pred_orphans"], "of": tally["pred_total"],
            "rate": rate(tally["pred_orphans"], tally["pred_total"])},
        "annotator_instances_orphaned_by_the_detector": {
            "count": tally["truth_orphans"], "of": tally["truth_total"],
            "rate": rate(tally["truth_orphans"], tally["truth_total"])},
        "annotator_instances_orphaned_by_another_annotator": {
            "count": tally["human_orphans"], "of": tally["human_total"],
            "rate": rate(tally["human_orphans"], tally["human_total"])},
        "orphaned_predictions_that_another_annotator_drew": tally["pred_orphans_rescued"],
        "orphaned_truths_that_no_other_annotator_drew": tally["truth_orphans_rescued"],
        "raw": tally,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
