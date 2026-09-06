"""Where the 456 false positives and 480 false negatives actually come from.

Three post-processing experiments have now been run against this error without
first asking what it is made of. Threshold tuning, a calibrated emission gate and
a boundary refiner all target one failure mode each, and all three assumed the
mode they targeted was the dominant one.

This asks the question directly. Every unmatched prediction and every unmatched
truth in the validation set is classified by its best overlap with the other
side, and each class is converted to the PQ it would return if that class were
eliminated. The classes are mutually exclusive, so the numbers say which lever is
worth building and which is worth abandoning.

Runs on cached candidates -- no detector inference, no GPU.
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

# The shipped operating point (exp_005 sweep, PQ 0.4404).
CONF = 0.35
MIN_AREA = 300
GROW = -1

# Overlap bands for the taxonomy. A prediction above MATCH is a true positive by
# definition, so the bands below it partition the failures.
MATCH = 0.5
NEAR = 0.25
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
    """The shipped one-pixel trim. Same structuring element as `predict.py`."""
    if iterations >= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(iterations) + 1,) * 2)
    return cv2.erode(binary, kernel).astype(np.uint8)


def iou_matrix(preds: list[np.ndarray], truths: list[np.ndarray]) -> np.ndarray:
    """Dense IoU via RLE, which is far cheaper than 2048x2048 numpy per pair."""
    if not preds or not truths:
        return np.zeros((len(preds), len(truths)), dtype=np.float64)
    p = mask_util.encode(np.asfortranarray(np.stack(preds, axis=-1)))
    t = mask_util.encode(np.asfortranarray(np.stack(truths, axis=-1)))
    return mask_util.iou(p, t, [0] * len(truths))


def classify(record_iou: np.ndarray) -> dict:
    """Split one record's predictions and truths into the failure taxonomy.

    A prediction and a truth match when their IoU exceeds 0.5. Because truths are
    disjoint and painted predictions are disjoint, that match is one-to-one and
    needs no assignment algorithm: two masks cannot each cover more than half of
    a third.
    """
    n_pred, n_truth = record_iou.shape
    matched_pred = np.zeros(n_pred, dtype=bool)
    matched_truth = np.zeros(n_truth, dtype=bool)
    pairs = []
    if n_pred and n_truth:
        for i, j in zip(*np.nonzero(record_iou > MATCH)):
            matched_pred[i] = True
            matched_truth[j] = True
            pairs.append((int(i), int(j), float(record_iou[i, j])))

    best_for_pred = record_iou.max(axis=1) if n_truth else np.zeros(n_pred)
    best_for_truth = record_iou.max(axis=0) if n_pred else np.zeros(n_truth)
    arg_for_pred = record_iou.argmax(axis=1) if n_truth else np.zeros(n_pred, int)
    arg_for_truth = record_iou.argmax(axis=0) if n_pred else np.zeros(n_truth, int)

    false_pos = []
    for i in range(n_pred):
        if matched_pred[i]:
            continue
        best = float(best_for_pred[i])
        target = int(arg_for_pred[i])
        false_pos.append({
            "index": i,
            "iou": best,
            # A miss whose best truth already belongs to another prediction is a
            # split: the detector broke one filament into pieces.
            "target_taken": bool(matched_truth[target]) if n_truth else False,
        })

    false_neg = []
    for j in range(n_truth):
        if matched_truth[j]:
            continue
        best = float(best_for_truth[j])
        source = int(arg_for_truth[j])
        false_neg.append({
            "index": j,
            "iou": best,
            # A truth whose best prediction already belongs to another truth is a
            # merge: the detector fused two filaments into one.
            "source_taken": bool(matched_pred[source]) if n_pred else False,
        })

    return {"pairs": pairs, "fp": false_pos, "fn": false_neg,
            "n_pred": n_pred, "n_truth": n_truth}


def band(iou: float) -> str:
    if iou >= NEAR:
        return "near"      # 0.25-0.50: boundary work could carry it over
    if iou >= TOUCH:
        return "graze"     # 0.10-0.25: overlaps, but the shape is wrong
    if iou > 0.0:
        return "sliver"    # touches something, essentially unrelated
    return "spurious"      # no overlap with any truth at all


def pq_from(sum_iou: float, tp: int, fp: int, fn: int) -> dict:
    denom = tp + 0.5 * fp + 0.5 * fn
    sq = sum_iou / tp if tp else 0.0
    rq = tp / denom if denom else 0.0
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


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

    painted_cache: dict[str, list[np.ndarray]] = {}

    def painted_for(stem: str) -> list[np.ndarray]:
        if stem not in painted_cache:
            candidates = []
            for entry in cached.get(stem, []):
                score = entry["features"]["score"]
                if score < CONF:
                    continue
                mask = erode(decode_rle(entry["counts"]), GROW)
                candidates.append((score, mask))
            painted_cache[stem] = [m for _, m, _ in paint_panoptic(candidates, MIN_AREA)]
        return painted_cache[stem]

    totals = {
        "sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0,
        "fp_band": defaultdict(int), "fn_band": defaultdict(int),
        "split_fp": 0, "merge_fn": 0,
        "fp_near_iou": [], "fn_near_iou": [],
        "fn_area": [], "tp_area": [],
    }
    records = 0

    for image_id in split.val_image_ids:
        annotations = by_image.get(image_id, [])
        truths = gt_masks(annotations)
        preds = painted_for(stem_of[image_id])
        if not truths and not preds:
            continue
        records += 1
        matrix = iou_matrix(preds, truths)
        result = classify(matrix)

        totals["tp"] += len(result["pairs"])
        totals["sum_iou"] += sum(p[2] for p in result["pairs"])
        totals["fp"] += len(result["fp"])
        totals["fn"] += len(result["fn"])
        for _, j, _ in result["pairs"]:
            totals["tp_area"].append(int(truths[j].sum()))
        for item in result["fp"]:
            totals["fp_band"][band(item["iou"])] += 1
            if item["target_taken"] and item["iou"] >= TOUCH:
                totals["split_fp"] += 1
            if band(item["iou"]) == "near":
                totals["fp_near_iou"].append(item["iou"])
        for item in result["fn"]:
            totals["fn_band"][band(item["iou"])] += 1
            if item["source_taken"] and item["iou"] >= TOUCH:
                totals["merge_fn"] += 1
            if band(item["iou"]) == "near":
                totals["fn_near_iou"].append(item["iou"])
            totals["fn_area"].append(int(truths[item["index"]].sum()))

    base = pq_from(totals["sum_iou"], totals["tp"], totals["fp"], totals["fn"])

    # Oracles. Each removes exactly one class and leaves the rest untouched, so
    # the deltas are the isolated value of solving that class.
    fp_band = dict(totals["fp_band"])
    fn_band = dict(totals["fn_band"])
    near_fp = fp_band.get("near", 0)
    near_fn = fn_band.get("near", 0)
    # A near miss is one prediction and one truth that failed to pair. Promoting
    # it converts one FP and one FN into a TP, and the pair enters SQ at the
    # threshold -- the least it can be worth.
    promoted = min(near_fp, near_fn)
    oracles = {
        "measured": base,
        "no_spurious_fp": pq_from(
            totals["sum_iou"], totals["tp"],
            totals["fp"] - fp_band.get("spurious", 0) - fp_band.get("sliver", 0),
            totals["fn"]),
        "near_misses_promoted": pq_from(
            totals["sum_iou"] + MATCH * promoted, totals["tp"] + promoted,
            totals["fp"] - promoted, totals["fn"] - promoted),
        "no_false_positives": pq_from(
            totals["sum_iou"], totals["tp"], 0, totals["fn"]),
        "no_false_negatives": pq_from(
            totals["sum_iou"], totals["tp"], totals["fp"], 0),
        "perfect_sq": pq_from(
            float(totals["tp"]), totals["tp"], totals["fp"], totals["fn"]),
    }

    report = {
        "operating_point": {"conf": CONF, "min_area": MIN_AREA, "grow": GROW},
        "records": records,
        "fp_taxonomy": fp_band,
        "fn_taxonomy": fn_band,
        "split_fp": totals["split_fp"],
        "merge_fn": totals["merge_fn"],
        "near_fp_iou_mean": float(np.mean(totals["fp_near_iou"])) if totals["fp_near_iou"] else None,
        "near_fn_iou_mean": float(np.mean(totals["fn_near_iou"])) if totals["fn_near_iou"] else None,
        "fn_area_median": float(np.median(totals["fn_area"])) if totals["fn_area"] else None,
        "tp_area_median": float(np.median(totals["tp_area"])) if totals["tp_area"] else None,
        "oracles": oracles,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
