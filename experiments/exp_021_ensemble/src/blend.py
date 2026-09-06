"""Average the two models' masks where both found the same filament.

exp_021 fused the two detectors at the level of which instances to emit, and
every rule only ever chose one model's mask or the other's. Blending them is a
different operation and the last one left that costs nothing.

exp_029 established that the annotated boundary is not recoverable from the
image, so anything trying to *find* the right edge is finished. Averaging is not
that. If the two models sit either side of the annotator's line with
uncorrelated error, their mean is closer to it than either — variance reduction,
which needs no information about where the line actually is.

With two binary masks the mean is degenerate: a pixel is in both, in neither, or
in exactly one. The disputed pixels are the boundary band, so the rules that
matter are how much of that band to keep — intersection, union, or a midpoint
taken from the distance transform, which is the closest thing to a true average
of two binary shapes.
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
CONF_A, GROW_A = 0.35, -1
CONF_B, GROW_B = 0.30, 0
MIN_AREA = 250
AGREE = 0.5


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


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shape halfway between two binary masks, by signed distance.

    Each mask is represented by the distance to its own boundary, negative
    outside and positive inside; averaging those fields and taking the zero level
    set gives a shape whose boundary lies midway between the two, which is what
    averaging two masks should mean.
    """
    union = ((a | b) > 0).astype(np.uint8)
    ys, xs = np.nonzero(union)
    if ys.size == 0:
        return a
    pad = 8
    top, bottom = max(0, ys.min() - pad), min(FULL, ys.max() + pad + 1)
    left, right = max(0, xs.min() - pad), min(FULL, xs.max() + pad + 1)

    def signed(mask: np.ndarray) -> np.ndarray:
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        return inside - outside

    pa = signed(a[top:bottom, left:right])
    pb = signed(b[top:bottom, left:right])
    out = np.zeros_like(a)
    out[top:bottom, left:right] = (((pa + pb) / 2.0) > 0).astype(np.uint8)
    return out


def load(path: str, conf: float, grow: int) -> dict[str, list[tuple[float, np.ndarray]]]:
    with open(path, encoding="utf-8") as fh:
        cached = json.load(fh)
    out = {}
    for stem, entries in cached.items():
        picked = [(e["features"]["score"], e["counts"]) for e in entries
                  if e["features"]["score"] >= conf]
        out[stem] = [(s, morph(decode_rle(c), grow)) for s, c in picked]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-a", required=True)
    parser.add_argument("--candidates-b", required=True)
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
    truths = {iid: gt_masks(anns) for iid, _, anns in records}

    a = load(args.candidates_a, CONF_A, GROW_A)
    b = load(args.candidates_b, CONF_B, GROW_B)
    print(f"a {sum(len(v) for v in a.values())} | b {sum(len(v) for v in b.values())}",
          flush=True)

    pairs: dict[str, list] = {}
    for stem in set(a) | set(b):
        ma, mb = a.get(stem, []), b.get(stem, [])
        if not ma:
            pairs[stem] = []
            continue
        if not mb:
            pairs[stem] = [(s, m, None) for s, m in ma]
            continue
        left = mask_util.encode(np.asfortranarray(np.stack([m for _, m in ma], -1)))
        right = mask_util.encode(np.asfortranarray(np.stack([m for _, m in mb], -1)))
        matrix = mask_util.iou(left, right, [0] * len(mb))
        best, arg = matrix.max(axis=1), matrix.argmax(axis=1)
        pairs[stem] = [(ma[i][0], ma[i][1], mb[int(arg[i])][1] if best[i] >= AGREE else None)
                       for i in range(len(ma))]

    matched = sum(1 for v in pairs.values() for _, _, partner in v if partner is not None)
    total = sum(len(v) for v in pairs.values())
    print(f"{matched}/{total} of A's masks have a partner in B at IoU {AGREE}", flush=True)

    def evaluate(name: str, combine) -> dict:
        rows = []
        for iid, stem, _ in records:
            candidates = [(s, m if partner is None else combine(m, partner))
                          for s, m, partner in pairs.get(stem, [])]
            painted = paint_panoptic(candidates, min_area=MIN_AREA) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], truths[iid]))
        row = aggregate_pq(rows)
        row["rule"] = name
        print(f"{name:<34} PQ {row['pq']:.4f}  SQ {row['sq']:.4f}  RQ {row['rq']:.4f}  "
              f"TP {row['tp']:4d} FP {row['fp']:4d} FN {row['fn']:4d}", flush=True)
        return row

    results = [
        evaluate("A alone (baseline)", lambda m, p: m),
        evaluate("intersection of the pair", lambda m, p: (m & p).astype(np.uint8)),
        evaluate("union of the pair", lambda m, p: (m | p).astype(np.uint8)),
        evaluate("distance-transform midpoint", midpoint),
    ]
    best = max(results, key=lambda r: r["pq"])
    print(f"\nbaseline {results[0]['pq']:.4f} -> best {best['pq']:.4f} "
          f"({best['pq'] - results[0]['pq']:+.4f})  [{best['rule']}]", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best, "baseline": results[0]}, fh, indent=2)


if __name__ == "__main__":
    main()
