"""Rank candidates by how sharply their mask field falls off, not only by confidence.

exp_016 found that 305 validation truths have a candidate in the pool that
matches them at IoU 0.5 and is thrown away by the confidence floor. Promoting
one earns a true positive and cancels a false negative; promoting a wrong one
costs half a false positive, so break-even is two wrong per one right --
precision above 33% against a base rate near 10.5%.

exp_005 already tried to beat confidence and lost, using mask area, elongation,
distance from the limb and solidity. Its verdict was that "the geometric features
add nothing beyond what confidence already encodes", and that is the right
reading: they describe the shape a mask happens to have, and confidence already
knows whether there is an object there.

This uses a different quantity. Detector confidence scores the *box*. The mask
comes from cutting a continuous field, built from the prototype basis, at logit
zero. How fast that field falls away from the cut is a statement about the mask
that confidence never sees: a real filament holds a strongly positive interior,
so raising the cut costs it little area, while a smear that barely crosses zero
collapses. exp_017 already computes the field at fourteen cuts for every
candidate, so the profile of areas across those cuts is free.

Evaluated out of fold by GroupKFold over photographs, and scored by PQ over
annotation records rather than over candidates -- the control that turned
exp_005's apparent win into a measured regression.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from shared.data_split import make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048
MATCH = 0.5
MIN_AREA = 300
SHIPPED_CONF = 0.35
SHIPPED_LEVEL = 0.0


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


def profile_features(areas: np.ndarray, score: float) -> np.ndarray:
    """One candidate's field profile, as ratios to the area at the shipped cut.

    Ratios rather than raw areas, so the descriptor is about the field's shape
    and not about how big the filament is -- size is already available, and is
    added once at the end in log space.
    """
    base = max(float(areas[0]), 1.0)
    ratios = areas / base
    return np.concatenate([ratios, [np.log1p(base), score]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True,
                        help="exp_017's level_candidates.json")
    parser.add_argument("--levels", required=True,
                        help="JSON list of the logit levels it was cached at")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    levels = json.loads(args.levels)
    shipped_index = levels.index(SHIPPED_LEVEL)

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}
    split = make_split(args.annotations)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]

    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    # Decode once. Every candidate at the shipped cut, plus its area profile.
    masks: dict[str, list[np.ndarray]] = {}
    scores: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for stem, entries in cached.items():
        stem_masks, stem_scores, stem_features = [], [], []
        for entry in entries:
            planes = [mask_util.decode({"size": [FULL, FULL],
                                        "counts": c.encode("ascii")})
                      for c in entry["counts"]]
            areas = np.array([float(p.sum()) for p in planes])
            # Order the profile from the shipped cut outward, so index 0 is the
            # denominator of every ratio.
            order = [shipped_index] + [i for i in range(len(levels)) if i != shipped_index]
            stem_masks.append(planes[shipped_index])
            stem_scores.append(entry["score"])
            stem_features.append(profile_features(areas[order], entry["score"]))
        masks[stem] = stem_masks
        scores[stem] = np.array(stem_scores)
        features[stem] = (np.stack(stem_features) if stem_features
                          else np.zeros((0, len(levels) + 2)))

    # Labels, one row per (candidate, annotation record). A candidate that two
    # annotators disagree about appears twice with different labels, which is the
    # honest representation of a label set that disagrees with itself.
    rows_x, rows_y, rows_group, rows_key = [], [], [], []
    for image_id, stem, annotations in records:
        truths = gt_masks(annotations)
        stem_masks = masks.get(stem, [])
        if not stem_masks:
            continue
        if truths:
            encoded_pred = mask_util.encode(np.asfortranarray(np.stack(stem_masks, axis=-1)))
            encoded_true = mask_util.encode(np.asfortranarray(np.stack(truths, axis=-1)))
            best = mask_util.iou(encoded_pred, encoded_true, [0] * len(truths)).max(axis=1)
        else:
            best = np.zeros(len(stem_masks))
        for index in range(len(stem_masks)):
            rows_x.append(features[stem][index])
            rows_y.append(int(best[index] >= MATCH))
            rows_group.append(stem)
            rows_key.append((image_id, stem, index))

    x = np.stack(rows_x)
    y = np.array(rows_y)
    groups = np.array(rows_group)
    print(f"{len(y)} candidate-record rows, {y.mean():.3%} match at IoU 0.5", flush=True)

    # Out of fold by photograph. A candidate must never be scored by a model that
    # has seen its own photograph under another annotator's labels.
    probability = np.zeros(len(y))
    for train_index, test_index in GroupKFold(n_splits=5).split(x, y, groups):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=2026)
        model.fit(x[train_index], y[train_index])
        probability[test_index] = model.predict_proba(x[test_index])[:, 1]

    by_key = {key: probability[i] for i, key in enumerate(rows_key)}

    # Does the profile add anything confidence does not? Fit the same model on
    # confidence alone and compare, rather than trusting that it must.
    confidence_only = np.zeros(len(y))
    score_column = x[:, -1:]
    for train_index, test_index in GroupKFold(n_splits=5).split(x, y, groups):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=2026)
        model.fit(score_column[train_index], y[train_index])
        confidence_only[test_index] = model.predict_proba(score_column[test_index])[:, 1]

    def average_precision(prob: np.ndarray) -> float:
        order = np.argsort(-prob)
        hits = y[order].cumsum()
        precision = hits / np.arange(1, len(y) + 1)
        return float((precision * y[order]).sum() / max(y.sum(), 1))

    quality = {
        "average_precision_field_profile": average_precision(probability),
        "average_precision_confidence_alone": average_precision(confidence_only),
        "average_precision_raw_confidence": average_precision(x[:, -1]),
    }
    print(json.dumps(quality, indent=2), flush=True)

    def evaluate(rule) -> dict:
        rows = []
        for image_id, stem, annotations in records:
            candidates = []
            for index, mask in enumerate(masks.get(stem, [])):
                if rule(scores[stem][index], by_key.get((image_id, stem, index), 0.0)):
                    candidates.append((float(scores[stem][index]), mask))
            painted = paint_panoptic(candidates, min_area=MIN_AREA) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], gt_masks(annotations)))
        return aggregate_pq(rows)

    results = []
    baseline = evaluate(lambda s, p: s >= SHIPPED_CONF)
    baseline.update({"rule": "confidence >= 0.35"})
    results.append(baseline)
    print(f"baseline {baseline['pq']:.4f}", flush=True)

    # Two families. A pure probability gate replaces confidence entirely; a
    # promotion rule keeps the shipped floor and only adds candidates beneath it,
    # which cannot lose anything the baseline already had.
    for gate in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        row = evaluate(lambda s, p, g=gate: p >= g)
        row.update({"rule": f"probability >= {gate}"})
        results.append(row)
        print(f"probability >= {gate:.2f}  PQ {row['pq']:.4f}  "
              f"TP {row['tp']} FP {row['fp']} FN {row['fn']}", flush=True)

    for gate in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        row = evaluate(lambda s, p, g=gate: s >= SHIPPED_CONF or p >= g)
        row.update({"rule": f"confidence >= 0.35 or probability >= {gate}"})
        results.append(row)
        print(f"promote at {gate:.2f}     PQ {row['pq']:.4f}  "
              f"TP {row['tp']} FP {row['fp']} FN {row['fn']}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    report = {"quality": quality, "sweep": results, "best": best,
              "baseline": baseline}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nbest: {json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
