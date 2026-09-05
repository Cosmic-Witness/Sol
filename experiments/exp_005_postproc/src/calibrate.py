"""Emit a prediction when it is likely to *match*, not when the detector is confident.

The decision rule
-----------------
Adding a prediction always adds 0.5 to the PQ denominator. It adds its IoU to
the numerator only if it matches a ground-truth filament at IoU > 0.5. So a
candidate is worth emitting exactly when

    P(match) * E[IoU | match]  >  0.5 * PQ

At the current operating point (PQ 0.44, E[IoU] 0.68) that is P(match) > 0.32.

Detector confidence is not P(match). It is trained to score the *box*, and a
box can be confidently placed around a filament whose mask still fails to reach
IoU 0.5 — which is precisely the near-miss population. Thresholding on
confidence therefore discards good masks with mediocre boxes and keeps bad masks
with good ones.

This fits P(match) directly from geometry that confidence does not see: mask
area, elongation, distance from the limb where contrast collapses, and solidity.

Honest evaluation
-----------------
The classifier is fitted and evaluated by grouped cross-validation over
*photographs*, never over candidates. Two candidates from one image share their
seeing, their limb darkening and often their filament; scoring a model on
candidates from a photograph it was fitted on would report a number that does
not survive contact with the test set. The reported PQ is always from folds the
classifier did not see.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

from shared.data_split import assert_disjoint, make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048
FEATURES = ("score", "log_area", "elongation", "limb", "solidity")


def gt_masks(annotations: list[dict]) -> list[np.ndarray]:
    out = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not segmentation or isinstance(segmentation, dict):
            continue
        rings = [r for r in segmentation if len(r) >= 6]
        if rings:
            rles = mask_util.frPyObjects(rings, FULL, FULL)
            out.append(mask_util.decode(mask_util.merge(rles)).astype(np.uint8))
    return out


def vectorise(features: dict) -> list[float]:
    return [
        features["score"],
        float(np.log1p(features["area"])),
        features["elongation"],
        features["limb"],
        features["solidity"],
    ]


def best_iou(candidate: np.ndarray, truths: list[np.ndarray]) -> float:
    if not truths:
        return 0.0
    encoded = mask_util.encode(np.asfortranarray(candidate))
    truth_rles = [mask_util.encode(np.asfortranarray(t)) for t in truths]
    ious = mask_util.iou([encoded], truth_rles, [0] * len(truth_rles))
    return float(np.max(ious)) if len(ious) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="npz written by nearmiss.py --dump-cache")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()

    import cv2
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}

    split = make_split(args.annotations)
    assert_disjoint(split)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]

    payload = json.loads(Path(args.cache).read_text())
    cache = {stem: entries for stem, entries in payload.items()}

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(args.grow) + 1,) * 2)

    def decoded(entry):
        mask = mask_util.decode({"size": entry["size"],
                                 "counts": entry["counts"].encode("ascii")})
        if args.grow:
            op = cv2.dilate if args.grow > 0 else cv2.erode
            mask = op(mask, kernel)
        return mask.astype(np.uint8)

    # Label every candidate by whether it actually matches, per record.
    rows, labels, groups, keys = [], [], [], []
    truths_by_record = {}
    for image_id, stem, annotations in records:
        truths = gt_masks(annotations)
        truths_by_record[image_id] = truths
        for position, entry in enumerate(cache.get(stem, [])):
            mask = decoded(entry)
            if int(mask.sum()) < args.min_area:
                continue
            rows.append(vectorise(entry["features"]))
            labels.append(1 if best_iou(mask, truths) > 0.5 else 0)
            groups.append(stem)
            keys.append((image_id, position))

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    print(f"{len(y)} candidates, {y.mean() * 100:.1f}% match at IoU>0.5", flush=True)

    # Out-of-fold probabilities, grouped by photograph.
    probability = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups=groups):
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                               learning_rate=0.06, random_state=2026)
        model.fit(X[train_idx], y[train_idx])
        probability[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    lookup = {k: probability[i] for i, k in enumerate(keys)}

    def evaluate(threshold: float, use_probability: bool) -> dict:
        per_image = []
        for image_id, stem, _annotations in records:
            candidates = []
            for position, entry in enumerate(cache.get(stem, [])):
                key = (image_id, position)
                if key not in lookup:
                    continue
                gate = lookup[key] if use_probability else entry["features"]["score"]
                if gate < threshold:
                    continue
                candidates.append((float(lookup[key]), decoded(entry)))
            painted = paint_panoptic(candidates, min_area=args.min_area) if candidates else []
            per_image.append(compute_pq([m for _, m, _ in painted], truths_by_record[image_id]))
        return aggregate_pq(per_image)

    results = []
    print(f"\n{'gate':>12}{'thresh':>8}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    for threshold in (0.20, 0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50):
        record = evaluate(threshold, use_probability=True)
        record.update(gate="probability", threshold=threshold)
        results.append(record)
        print(f"{'probability':>12}{threshold:>8.2f}{record['pq']:>9.4f}{record['sq']:>8.4f}"
              f"{record['rq']:>8.4f}{record['tp']:>7}{record['fp']:>7}{record['fn']:>7}", flush=True)
    for threshold in (0.30, 0.35, 0.40):
        record = evaluate(threshold, use_probability=False)
        record.update(gate="confidence", threshold=threshold)
        results.append(record)
        print(f"{'confidence':>12}{threshold:>8.2f}{record['pq']:>9.4f}{record['sq']:>8.4f}"
              f"{record['rq']:>8.4f}{record['tp']:>7}{record['fp']:>7}{record['fn']:>7}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nBEST {best['gate']} @ {best['threshold']}: PQ {best['pq']:.4f}", flush=True)
    Path(args.out).write_text(json.dumps({"sweep": results, "best": best}, indent=2))


if __name__ == "__main__":
    main()
