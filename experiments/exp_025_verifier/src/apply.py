"""Score validation candidates with the verifier and sweep the emission rule.

The arithmetic this is chasing, from exp_016 and the break-even in exp_019:
promoting k correct candidates and m incorrect ones from below the confidence
floor moves PQ to (578.2 + 0.65k) / (1313 + 0.5k + 0.5m). Break-even is two wrong
per one right. 305 validation truths have a covering candidate that confidence
discards, and taking them all at 80% precision is worth about +0.076 PQ.

Two families of rule are swept, and the difference matters. A pure verifier gate
replaces confidence entirely and can lose detections the baseline already had. A
promotion rule keeps the confidence floor and only adds candidates beneath it, so
it cannot lose ground -- it can only fail to gain.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

from experiments.exp_025_verifier.src.harvest import CROP, crop_of
from experiments.exp_025_verifier.src.model import Verifier
from shared.data_split import make_split
from shared.utils import aggregate_pq, compute_pq, decode_rle, paint_panoptic

FULL = 2048
BASE_CONF = 0.30      # exp_022's settled floor for this detector
MIN_AREA = 250
GROW = 0


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
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from pathlib import Path
    images_dir = Path(args.images)

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}
    split = make_split(args.annotations)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]
    truths = {iid: gt_masks(anns) for iid, _, anns in records}

    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    checkpoint = torch.load(args.verifier, map_location="cpu")
    model = Verifier(checkpoint.get("width", 24))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    print(f"verifier loaded (val AP {checkpoint.get('val_ap')})", flush=True)

    # Score every candidate once. Masks stay as RLE between uses.
    pool: dict[str, list[tuple[float, float, bytes]]] = {}
    for position, (stem, entries) in enumerate(sorted(cached.items()), start=1):
        photograph = cv2.imread(str(images_dir / stem), cv2.IMREAD_GRAYSCALE)
        if photograph is None:
            continue
        batch, keep = [], []
        for entry in entries:
            mask = decode_rle(entry["counts"])
            cropped = crop_of(photograph, mask)
            if cropped is None:
                continue
            patch, patch_mask = cropped
            batch.append(np.stack([patch.astype(np.float32) / 255.0,
                                   patch_mask.astype(np.float32)]))
            keep.append(entry)
        if not batch:
            pool[stem] = []
            continue
        with torch.no_grad():
            logits = model(torch.from_numpy(np.stack(batch))).numpy()
        probability = 1.0 / (1.0 + np.exp(-logits))
        pool[stem] = [(e["features"]["score"], float(p), e["counts"])
                      for e, p in zip(keep, probability)]
        if position % 25 == 0 or position == len(cached):
            print(f"  scored {position}/{len(cached)}", flush=True)

    def evaluate(rule) -> dict:
        rows = []
        for iid, stem, _ in records:
            candidates = []
            for conf, prob, counts in pool.get(stem, []):
                emitted = rule(conf, prob)
                if emitted is not None:
                    mask = decode_rle(counts)
                    if GROW:
                        k = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * abs(GROW) + 1,) * 2)
                        mask = (cv2.dilate if GROW > 0 else cv2.erode)(mask, k)
                    candidates.append((emitted, mask))
            painted = paint_panoptic(candidates, min_area=MIN_AREA) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], truths[iid]))
        return aggregate_pq(rows)

    results = []

    def record(name: str, rule) -> dict:
        row = evaluate(rule)
        row["rule"] = name
        results.append(row)
        print(f"{name:<44} PQ {row['pq']:.4f}  SQ {row['sq']:.4f}  RQ {row['rq']:.4f}  "
              f"TP {row['tp']:4d} FP {row['fp']:4d} FN {row['fn']:4d}", flush=True)
        return row

    baseline = record(f"confidence >= {BASE_CONF} (baseline)",
                      lambda c, p: c if c >= BASE_CONF else None)

    # Promotion: keep the floor, add what the verifier vouches for beneath it.
    for gate in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        record(f"confidence >= {BASE_CONF} or verifier >= {gate}",
               lambda c, p, g=gate: (c if c >= BASE_CONF else
                                     (BASE_CONF - 0.01 if p >= g else None)))

    # Pure verifier gate, ordering by the verifier rather than by confidence.
    for gate in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        record(f"verifier >= {gate} alone",
               lambda c, p, g=gate: (p if p >= g else None))

    best = max(results, key=lambda r: r["pq"])
    print(f"\nbaseline {baseline['pq']:.4f} -> best {best['pq']:.4f} "
          f"({best['pq'] - baseline['pq']:+.4f})  [{best['rule']}]", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best, "baseline": baseline}, fh, indent=2)


if __name__ == "__main__":
    main()
