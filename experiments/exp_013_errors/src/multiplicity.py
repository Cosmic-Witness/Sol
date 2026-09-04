"""Is validation optimistic because it is weighted towards agreed-upon photographs?

Validation PQ is averaged over 180 annotation records drawn from 106
photographs, because 47 of them were labelled independently by two or three
people and each labelling is its own record. The test set is 180 photographs
scored once each.

That weighting is not neutral if multiply-annotated photographs are easier. A
photograph two experts both chose to label carefully, and agreed on, is
plausibly one with clear unambiguous filaments -- and it enters the validation
average two or three times.

The train and test sets are otherwise well matched: by year the two histograms
agree to a couple of points except at 2021-2022, where test carries about twice
the share of a small count, and by GONG station they agree to within three
points. So distribution shift does not explain the 0.078 gap between validation
and the leaderboard, and this is the remaining structural candidate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util

from shared.data_split import make_split
from shared.utils import aggregate_pq, compute_pq, decode_rle, paint_panoptic

FULL = 2048
CONF = 0.35
MIN_AREA = 300
GROW = -1


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
    ids_of_stem = defaultdict(list)
    for record in coco["images"]:
        ids_of_stem[record["file_name"]].append(record["id"])

    split = make_split(args.annotations)
    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(GROW) + 1,) * 2)
    painted: dict[str, list[np.ndarray]] = {}
    for stem in split.val_stems:
        candidates = []
        for entry in cached.get(stem, []):
            score = entry["features"]["score"]
            if score < CONF:
                continue
            mask = cv2.erode(decode_rle(entry["counts"]), kernel).astype(np.uint8)
            candidates.append((score, mask))
        painted[stem] = [m for _, m, _ in paint_panoptic(candidates, MIN_AREA)]

    per_record: dict[str, list[dict]] = defaultdict(list)
    for image_id in split.val_image_ids:
        stem = stem_of[image_id]
        row = compute_pq(painted.get(stem, []), gt_masks(by_image.get(image_id, [])))
        per_record[stem].append(row)

    single = [r for stem, rows in per_record.items()
              if len(ids_of_stem[stem]) == 1 for r in rows]
    multiple = [r for stem, rows in per_record.items()
                if len(ids_of_stem[stem]) > 1 for r in rows]
    everything = [r for rows in per_record.values() for r in rows]

    # Per photograph: pool a photograph's records into one contribution, so a
    # thrice-labelled photograph counts once, as it would in the test set.
    per_photograph = []
    for stem, rows in per_record.items():
        pooled = {key: sum(row[key] for row in rows) / len(rows)
                  for key in ("iou_sum", "tp", "fp", "fn")} if rows else None
        if pooled:
            per_photograph.append(pooled)

    report = {
        "records": {
            "all": aggregate_pq(everything),
            "singly_annotated": aggregate_pq(single) if single else None,
            "multiply_annotated": aggregate_pq(multiple) if multiple else None,
        },
        "counts": {
            "photographs": len(per_record),
            "singly_annotated_records": len(single),
            "multiply_annotated_records": len(multiple),
        },
        "per_photograph_weighting": aggregate_pq(per_photograph),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
