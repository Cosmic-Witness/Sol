"""Measure thin-instance survival on the canonical TRAIN fold, without images.

COCO rasterization is exact; nearest-neighbor downsampling is a diagnostic,
not a claim about either model's learned output. One instance is decoded at a
time. No test annotations or dense dataset cache are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

from shared.data_split import make_split, assert_disjoint


def audit(path, resolutions, min_native_area=300):
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    split = make_split(str(path))
    assert_disjoint(split)
    train_ids = set(split.train_image_ids)
    records = {r["id"]: r for r in coco["images"]}
    counts = Counter(a["image_id"] for a in coco["annotations"] if a["image_id"] in train_ids)
    totals = {str(r): {"erased": 0, "below_scaled_area": 0, "iou_sum": 0.0} for r in resolutions}
    n = 0
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in train_ids:
            continue
        record = records[annotation["image_id"]]
        h, w = record["height"], record["width"]
        seg = annotation["segmentation"]
        if isinstance(seg, list):
            rle = mask_util.merge(mask_util.frPyObjects(seg, h, w))
        elif isinstance(seg["counts"], list):
            rle = mask_util.frPyObjects(seg, h, w)
        else:
            rle = seg
        mask = mask_util.decode(rle)
        if not mask.any():
            raise ValueError(f"empty annotation: {annotation['id']}")
        n += 1
        for resolution in resolutions:
            small = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
            restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            item = totals[str(resolution)]
            item["erased"] += int(not small.any())
            # Scale area to the SAME physical threshold; 300 coarse pixels
            # would exclude much larger physical filaments than 300 native pixels.
            item["below_scaled_area"] += int(small.sum() < min_native_area * resolution**2 / (h * w))
            item["iou_sum"] += float((mask & restored).sum() / (mask | restored).sum())
    for item in totals.values():
        item["mean_roundtrip_iou"] = item.pop("iou_sum") / n
    sizes = [counts[i] for i in split.train_image_ids]
    return {
        "annotation_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "split_seed": 2026, "fold": "train", "split": split.summary(),
        "instances": n, "min_native_area": min_native_area,
        "instances_per_record": {"max": max(sizes), "p95": float(np.percentile(sizes, 95)),
                                 "median": float(np.median(sizes))},
        "crowded_record_ids": sorted(counts, key=counts.get, reverse=True)[:5],
        "resolutions": totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[800, 1024, 1280, 1536, 2048])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.resolutions) <= 0:
        parser.error("resolutions must be positive")
    result = audit(args.annotations, args.resolutions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
