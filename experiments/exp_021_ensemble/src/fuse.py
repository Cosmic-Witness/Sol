"""Fuse two detectors whose errors point in opposite directions.

Every correction tried so far has been one model's output rewritten by a rule or
by a second network, and all of them failed for the same measured reason: the
detector's remaining error is not predictable from the detector's own output.

Two detectors are a different mechanism. exp_002 trained at 1280 on fat targets;
exp_010 trained at 2048 on rasterisation-corrected ones. At their own best
operating points they fail in opposite directions:

    exp_002   TP 845   FP 456   FN 480      recall, bought with false positives
    exp_010   TP 766   FP 338   FN 559      precision, bought with misses

If those errors were the same errors, the pair is worth nothing. If they are
decorrelated, then a candidate both models propose is far more likely to be real
than its own confidence says, and a candidate only one proposes is less. That is
information neither model has about itself, which is precisely what the
single-model re-rankers lacked.

Runs on the cached candidate sets both models already wrote. No inference.
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
MIN_AREA = 300
AGREE_IOU = 0.5


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


def erode(binary: np.ndarray, pixels: int) -> np.ndarray:
    if not pixels:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(binary, kernel).astype(np.uint8)


def iou_matrix(left: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)))
    a = mask_util.encode(np.asfortranarray(np.stack(left, axis=-1)))
    b = mask_util.encode(np.asfortranarray(np.stack(right, axis=-1)))
    return mask_util.iou(a, b, [0] * len(right))


def load(path: str, grow: int) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Decode a cached candidate set once, at the erosion its model prefers."""
    with open(path, encoding="utf-8") as fh:
        cached = json.load(fh)
    out: dict[str, list[tuple[float, np.ndarray]]] = {}
    for stem, entries in cached.items():
        out[stem] = [(e["features"]["score"], erode(decode_rle(e["counts"]), grow))
                     for e in entries]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-a", required=True, help="exp_002 cache")
    parser.add_argument("--candidates-b", required=True, help="exp_010 cache")
    parser.add_argument("--grow-a", type=int, default=-1)
    parser.add_argument("--grow-b", type=int, default=0)
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

    a = load(args.candidates_a, args.grow_a)
    b = load(args.candidates_b, args.grow_b)
    print(f"a: {sum(len(v) for v in a.values())} candidates | "
          f"b: {sum(len(v) for v in b.values())} candidates", flush=True)

    # For every candidate of each model, how strongly the other model agrees:
    # the best IoU against any of the other's candidates, and that partner's
    # confidence. Computed once at the widest floor, then filtered per sweep row.
    agreement: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for stem in set(a) | set(b):
        ma = [m for _, m in a.get(stem, [])]
        mb = [m for _, m in b.get(stem, [])]
        matrix = iou_matrix(ma, mb)
        if matrix.size:
            best_a, best_b = matrix.max(axis=1), matrix.max(axis=0)
            partner_a = np.array([b[stem][j][0] for j in matrix.argmax(axis=1)])
            partner_b = np.array([a[stem][i][0] for i in matrix.argmax(axis=0)])
        else:
            best_a = np.zeros(len(ma)); best_b = np.zeros(len(mb))
            partner_a = np.zeros(len(ma)); partner_b = np.zeros(len(mb))
        agreement[stem] = (best_a, partner_a, best_b, partner_b)

    def evaluate(rule) -> dict:
        rows = []
        for _iid, stem, annotations in records:
            best_a, partner_a, best_b, partner_b = agreement[stem]
            candidates = []
            for i, (score, mask) in enumerate(a.get(stem, [])):
                s = rule("a", score, float(best_a[i]), float(partner_a[i]))
                if s is not None:
                    candidates.append((s, mask))
            for j, (score, mask) in enumerate(b.get(stem, [])):
                s = rule("b", score, float(best_b[j]), float(partner_b[j]))
                if s is not None:
                    candidates.append((s, mask))
            painted = paint_panoptic(candidates, min_area=MIN_AREA) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], gt_masks(annotations)))
        return aggregate_pq(rows)

    results = []

    def record(name: str, rule) -> dict:
        row = evaluate(rule)
        row["rule"] = name
        results.append(row)
        print(f"{name:<52} PQ {row['pq']:.4f}  SQ {row['sq']:.4f}  "
              f"RQ {row['rq']:.4f}  TP {row['tp']:4d} FP {row['fp']:4d} FN {row['fn']:4d}",
              flush=True)
        return row

    # The two models alone, at their own operating points, as the bar to clear.
    record("A alone (exp_002, conf 0.35)",
           lambda m, s, agree, partner: s if (m == "a" and s >= 0.35) else None)
    record("B alone (exp_010, conf 0.30)",
           lambda m, s, agree, partner: s if (m == "b" and s >= 0.30) else None)

    # Naive union: everything both models emit, painted by confidence. The
    # control that shows whether fusion needs to be selective at all.
    record("union at each model's own floor",
           lambda m, s, agree, partner:
               s if (m == "a" and s >= 0.35) or (m == "b" and s >= 0.30) else None)

    # Agreement gating. A candidate the other model also found is promoted below
    # its own floor; one it did not find must clear a higher bar alone.
    for solo_a, solo_b, floor in ((0.45, 0.40, 0.20), (0.50, 0.45, 0.20),
                                  (0.45, 0.40, 0.15), (0.55, 0.50, 0.25)):
        def rule(m, s, agree, partner, solo_a=solo_a, solo_b=solo_b, floor=floor):
            solo = solo_a if m == "a" else solo_b
            if agree >= AGREE_IOU and partner >= floor:
                return s + 1.0          # agreed: paint first, keep below floor
            base = 0.35 if m == "a" else 0.30
            return s if s >= solo and s >= base else None
        record(f"agreed, or alone above {solo_a}/{solo_b} (partner >= {floor})", rule)

    # A as the backbone, B only used to veto: A's candidates that B contradicts
    # by finding nothing there must clear a higher confidence.
    for veto in (0.45, 0.50, 0.55):
        def rule(m, s, agree, partner, veto=veto):
            if m != "a" or s < 0.35:
                return None
            if agree >= AGREE_IOU:
                return s + 1.0
            return s if s >= veto else None
        record(f"A only, unconfirmed needs {veto}", rule)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nbest: {json.dumps(best, indent=2)}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best}, fh, indent=2)


if __name__ == "__main__":
    main()
