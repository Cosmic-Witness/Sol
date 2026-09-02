"""Attack the near-miss population: predictions that are real filaments scored just under IoU 0.5.

The arithmetic that motivates this
----------------------------------
Under PQ a prediction that fails to match costs 0.5 of denominator as a false
positive, and the ground truth it failed to claim costs another 0.5 as a false
negative. Converting that pair into a true positive replaces 1.0 of denominator
with 1.0 of denominator — the denominator does not move — while the numerator
gains the full IoU. Near-misses are therefore the cheapest true positives
available: they need no new detections, only slightly better masks.

At 2048 the model sits at TP 854, FP 635, FN 471 with SQ 0.6695. Converting a
third of the false positives would lift RQ from 0.607 to about 0.757 and PQ to
roughly 0.50, without detecting a single new filament.

Two levers are swept here, both post-hoc and both CPU-only.

**Mask dilation.** Ultralytics thresholds its mask prototypes at 0.5, which for
structures a few pixels wide systematically erodes barbs — the prototype field
is smooth and a thin ridge loses its tails first. Growing the mask by a pixel or
two approximates a lower threshold. It should help near-misses and hurt masks
that already match, so the optimum is an empirical trade rather than a direction.

**Calibrated emission.** Adding any prediction adds 0.5 to the denominator
whether or not it matches, and adds its IoU to the numerator only if it does. So
a candidate is worth emitting when P(match) * E[IoU] exceeds 0.5 * PQ — about a
0.30 probability at the current score. Detector confidence is not that
probability: it is calibrated for the box, not for whether the mask will clear
an IoU threshold. A small model over confidence, area, elongation and limb
distance estimates it directly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

from shared.data_split import assert_disjoint, make_split
from shared.solar_disk import disk_mask, find_disk
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048


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


def instance_features(mask: np.ndarray, score: float, centre: tuple[float, float],
                      radius: float) -> dict:
    """Cheap geometric descriptors that plausibly predict whether a mask will match."""
    area = float(mask.sum())
    ys, xs = np.nonzero(mask)
    if area < 1:
        return {"score": score, "area": 0.0, "elongation": 0.0, "limb": 0.0, "solidity": 0.0}

    # Elongation via the second-moment ratio: filaments are long and thin, and a
    # blob that is not elongated is more often a sunspot or an artefact.
    x_var, y_var = xs.var(), ys.var()
    xy_cov = float(((xs - xs.mean()) * (ys - ys.mean())).mean())
    trace = x_var + y_var
    det = x_var * y_var - xy_cov ** 2
    disc = max(trace * trace / 4 - det, 0.0) ** 0.5
    major, minor = trace / 2 + disc, max(trace / 2 - disc, 1e-6)
    elongation = float((major / minor) ** 0.5)

    # Distance from disk centre in radii: contrast falls towards the limb, so
    # masks there are systematically worse.
    limb = float(np.hypot(xs.mean() - centre[0], ys.mean() - centre[1]) / max(radius, 1))

    hull_area = area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(np.vstack(contours))
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
    return {"score": float(score), "area": area, "elongation": elongation,
            "limb": limb, "solidity": float(area / hull_area)}


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(mask, kernel).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--floor-conf", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}

    split = make_split(args.annotations)
    assert_disjoint(split)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]
    photographs = sorted({s for _, s, _ in records})
    print(f"{len(records)} records over {len(photographs)} photographs", flush=True)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_dir = Path(args.images)

    # One inference pass. Everything below re-scores this cache.
    cache: dict[str, list] = {}
    for position, stem in enumerate(photographs, start=1):
        path = images_dir / stem
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        cx, cy, radius = find_disk(raw)

        result = model.predict(str(path), imgsz=args.imgsz, conf=args.floor_conf,
                               iou=0.60, max_det=100, retina_masks=True, verbose=False)[0]
        entries = []
        if result.masks is not None and len(result.masks.data):
            data = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for index in range(len(data)):
                mask = data[index].astype(np.uint8)
                if mask.shape != (FULL, FULL):
                    mask = cv2.resize(mask, (FULL, FULL), interpolation=cv2.INTER_NEAREST)
                if mask.sum() < 40:
                    continue
                entries.append({
                    "rle": mask_util.encode(np.asfortranarray(mask)),
                    "features": instance_features(mask, scores[index], (cx, cy), radius),
                })
        cache[stem] = entries
        if position % 20 == 0 or position == len(photographs):
            print(f"  inferred {position}/{len(photographs)}", flush=True)

    total = sum(len(v) for v in cache.values())
    print(f"cached {total} candidates\n", flush=True)

    def evaluate(conf: float, min_area: int, grow: int) -> dict:
        rows = []
        for _iid, stem, annotations in records:
            candidates = []
            for entry in cache.get(stem, []):
                if entry["features"]["score"] < conf:
                    continue
                mask = mask_util.decode(entry["rle"])
                if grow:
                    mask = dilate(mask, grow)
                candidates.append((entry["features"]["score"], mask))
            painted = paint_panoptic(candidates, min_area=min_area) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], gt_masks(annotations)))
        return aggregate_pq(rows)

    results = []
    print(f"{'conf':>6}{'min_area':>10}{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    # The first grid put the optimum at conf 0.35 / grow -1, both on its
    # boundary, which means the true optimum lies outside it. Dilation was
    # monotonically catastrophic (PQ 0.19 at +3), so the grid extends into
    # erosion and higher confidence only.
    for conf in (0.30, 0.35, 0.40, 0.45, 0.50):
        for grow in (-4, -3, -2, -1, 0):
            record = evaluate(conf, 300, grow)
            record.update(conf=conf, min_area=300, grow=grow)
            results.append(record)
            print(f"{conf:>6.2f}{300:>10}{grow:>6}{record['pq']:>9.4f}{record['sq']:>8.4f}"
                  f"{record['rq']:>8.4f}{record['tp']:>7}{record['fp']:>7}{record['fn']:>7}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nBEST conf={best['conf']} grow={best['grow']} PQ={best['pq']:.4f} "
          f"(SQ {best['sq']:.4f} x RQ {best['rq']:.4f})", flush=True)
    Path(args.out).write_text(json.dumps({"sweep": results, "best": best}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
