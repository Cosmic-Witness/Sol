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

    # Masks stay compressed between uses. Holding 2114 dense 2048-square arrays
    # is nine gigabytes per erosion level, and caching three levels killed the
    # first attempt with SIGKILL. An RLE of a mostly-empty mask is a couple of
    # kilobytes, so the whole pool fits in a few megabytes and each evaluation
    # decodes only the candidates it actually paints.
    raw = {stem: [(e["features"]["score"], e["counts"]) for e in entries]
           for stem, entries in cached.items()}
    print(f"{sum(len(v) for v in raw.values())} candidates over "
          f"{len(raw)} photographs", flush=True)

    truths = {iid: gt_masks(anns) for iid, _, anns in records}

    def encode(binary: np.ndarray) -> bytes:
        return mask_util.encode(np.asfortranarray(binary))["counts"]

    def rles_for(grow: int) -> dict[str, list[tuple[float, bytes]]]:
        if grow == 0:
            return {stem: [(sc, c.encode("ascii") if isinstance(c, str) else c)
                           for sc, c in v] for stem, v in raw.items()}
        out = {}
        for stem, entries in raw.items():
            out[stem] = [(sc, encode(morph(decode_rle(c), grow))) for sc, c in entries]
        return out

    results = []
    header = (f"{'conf':>6}{'area':>7}{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}"
              f"{'TP':>6}{'FP':>6}{'FN':>6}")
    print("\n" + header, flush=True)

    # grow -1 already measured far worse (0.4020 against 0.4226) and conf 0.30
    # already confirmed interior, so this pass asks the one question the first
    # attempt died before reaching: whether the corrected targets want the mask
    # grown rather than trimmed.
    for grow in (0, 1, 2):
        pool = rles_for(grow)
        for conf in (0.25, 0.30, 0.35):
            for min_area in (250, 300, 400):
                rows = []
                for iid, stem, _ in records:
                    picked = [(sc, mask_util.decode({"size": [FULL, FULL], "counts": c}))
                              for sc, c in pool.get(stem, []) if sc >= conf]
                    painted = (paint_panoptic(picked, min_area=min_area)
                               if picked else [])
                    rows.append(compute_pq([m for _, m, _ in painted], truths[iid]))
                row = aggregate_pq(rows)
                row.update({"conf": conf, "min_area": min_area, "grow": grow})
                results.append(row)
                print(f"{conf:6.2f}{min_area:7d}{grow:6d}{row['pq']:9.4f}"
                      f"{row['sq']:8.4f}{row['rq']:8.4f}"
                      f"{row['tp']:6d}{row['fp']:6d}{row['fn']:6d}", flush=True)
        del pool

    best = max(results, key=lambda r: r["pq"])
    interior = best["grow"] != 2
    print(f"\nbest: {json.dumps(best, indent=2)}")
    print(f"optimum is {'interior' if interior else 'ON THE BOUNDARY -- widen again'}",
          flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best, "interior": interior}, fh, indent=2)


if __name__ == "__main__":
    main()
