"""Widen the operating-point grid past the edge the last one stopped at.

exp_015 fixed exp_010's operating point with a grid of conf {0.30, 0.35, 0.40}
crossed with grow {0, -1}, at a fixed 300-pixel area floor. It chose conf 0.30
and grow 0 -- the lowest confidence and the largest mask in the grid. **Both on
the boundary**, which means the optimum is outside it, and exp_005 recorded that
exact lesson about its own first grid before repeating it here.

The direction is predictable from what the model is. Trained on
rasterisation-corrected targets, its masks are thinner than the 1280-trained
model's, so where that one wanted a pixel eroded this one may want a pixel added.
And its precision is far higher at the same confidence -- 338 false positives
against 456 -- so it can afford to emit more.

Runs on the candidate cache exp_015 already dumped. No inference, no GPU.
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


def morph(binary: np.ndarray, pixels: int) -> np.ndarray:
    if not pixels:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(binary, kernel).astype(np.uint8)


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
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]

    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    # Decode once per (stem, grow); the grid revisits each many times.
    decoded = {stem: [(e["features"]["score"], decode_rle(e["counts"]))
                      for e in entries]
               for stem, entries in cached.items()}
    print(f"{sum(len(v) for v in decoded.values())} candidates over "
          f"{len(decoded)} photographs", flush=True)

    truths = {iid: gt_masks(anns) for iid, _, anns in records}

    results = []
    header = (f"{'conf':>6}{'area':>7}{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}"
              f"{'TP':>6}{'FP':>6}{'FN':>6}")

    grown_cache: dict[int, dict] = {}

    def grown_for(grow: int) -> dict:
        if grow not in grown_cache:
            grown_cache[grow] = {stem: [(s, morph(m, grow)) for s, m in v]
                                 for stem, v in decoded.items()}
        return grown_cache[grow]

    def score(conf: float, min_area: int, grow: int) -> dict:
        grown = grown_for(grow)
        rows = []
        for iid, stem, _ in records:
            candidates = [(s, m) for s, m in grown.get(stem, []) if s >= conf]
            painted = (paint_panoptic(candidates, min_area=min_area)
                       if candidates else [])
            rows.append(compute_pq([m for _, m, _ in painted], truths[iid]))
        row = aggregate_pq(rows)
        row.update({"conf": conf, "min_area": min_area, "grow": grow})
        results.append(row)
        print(f"{conf:6.2f}{min_area:7d}{grow:6d}{row['pq']:9.4f}"
              f"{row['sq']:8.4f}{row['rq']:8.4f}"
              f"{row['tp']:6d}{row['fp']:6d}{row['fn']:6d}", flush=True)
        return row

    # Coarse first, then refine around its winner. A full product of every
    # candidate value is 120 evaluations of 180 records each and takes hours; the
    # surface is smooth enough that two passes find the same optimum in a
    # fraction of that, and the boundary check below still catches a grid that
    # stopped too early.
    print("\n== coarse ==\n" + header, flush=True)
    for grow in (-1, 0, 1, 2):
        for conf in (0.15, 0.25, 0.35):
            for min_area in (200, 300):
                score(conf, min_area, grow)

    coarse = max(results, key=lambda r: r["pq"])
    print(f"\ncoarse winner: conf {coarse['conf']} area {coarse['min_area']} "
          f"grow {coarse['grow']} PQ {coarse['pq']:.4f}", flush=True)

    confs = sorted({round(max(0.05, coarse["conf"] + d), 2)
                    for d in (-0.05, 0.0, 0.05)})
    areas = sorted({max(100, coarse["min_area"] + d) for d in (-100, -50, 0, 50, 100)})
    grows = sorted({coarse["grow"] + d for d in (-1, 0, 1)})
    print("\n== refine ==\n" + header, flush=True)
    seen = {(r["conf"], r["min_area"], r["grow"]) for r in results}
    for grow in grows:
        for conf in confs:
            for min_area in areas:
                if (conf, min_area, grow) not in seen:
                    score(conf, min_area, grow)

    best = max(results, key=lambda r: r["pq"])
    tested_conf = sorted({r["conf"] for r in results})
    tested_area = sorted({r["min_area"] for r in results})
    tested_grow = sorted({r["grow"] for r in results})
    interior = (best["conf"] not in (tested_conf[0], tested_conf[-1])
                and best["min_area"] not in (tested_area[0], tested_area[-1])
                and best["grow"] not in (tested_grow[0], tested_grow[-1]))
    print(f"\nbest: {json.dumps(best, indent=2)}")
    print(f"optimum is {'interior' if interior else 'ON THE BOUNDARY -- widen again'}",
          flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best, "interior": interior}, fh, indent=2)


if __name__ == "__main__":
    main()
