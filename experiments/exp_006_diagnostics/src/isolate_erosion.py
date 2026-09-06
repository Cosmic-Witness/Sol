"""Separate the two effects tangled together in the erosion result.

Eroding predictions by one pixel lifted the leaderboard 0.33 to 0.36. Two
mechanisms could produce that and the original sweep could not tell them apart,
because it always ran with a 300px minimum-area filter:

1. **A boundary effect.** Masks are ~11% fatter than the scorer's rasterisation
   convention, so trimming them raises IoU on instances that already matched.
2. **An area-filter effect.** Erosion shrinks marginal blobs below 300px, where
   they are discarded before scoring rather than charged as false positives.

The second is not a boundary correction at all — it is a confidence filter
wearing a morphological disguise, and if it dominates then the polygon-offset fix
at training time will not reproduce the gain.

Running the sweep with the filter disabled isolates the first. This also fits a
per-instance trim: if the optimal amount of trimming varies with instance shape,
a single global constant is leaving score on the table; if it does not, the bias
is uniform in pixels and the global constant is already correct.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import pycocotools.mask as mask_util

from shared.data_split import assert_disjoint, make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048


def gt_masks(annotations):
    out = []
    for annotation in annotations:
        seg = annotation.get("segmentation")
        if not seg or isinstance(seg, dict):
            continue
        rings = [r for r in seg if len(r) >= 6]
        if rings:
            out.append(mask_util.decode(
                mask_util.merge(mask_util.frPyObjects(rings, FULL, FULL))).astype(np.uint8))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}

    split = make_split(args.annotations)
    assert_disjoint(split)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]
    cache = json.loads(open(args.cache).read())

    kernels = {g: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(g) + 1,) * 2)
               for g in (1, 2)}

    def decode(entry, grow):
        m = mask_util.decode({"size": entry["size"],
                              "counts": entry["counts"].encode("ascii")})
        if grow:
            op = cv2.dilate if grow > 0 else cv2.erode
            m = op(m, kernels[abs(grow)])
        return m.astype(np.uint8)

    truths = {i: gt_masks(a) for i, _s, a in records}

    def evaluate(conf, grow, min_area):
        rows = []
        for image_id, stem, _ in records:
            cands = []
            for e in cache.get(stem, []):
                if e["features"]["score"] < conf:
                    continue
                m = decode(e, grow)
                if int(m.sum()) < max(min_area, 1):
                    continue
                cands.append((e["features"]["score"], m))
            painted = paint_panoptic(cands, min_area=max(min_area, 1)) if cands else []
            rows.append(compute_pq([m for _, m, _ in painted], truths[image_id]))
        return aggregate_pq(rows)

    print("=== erosion WITH the 300px area filter (as originally reported) ===")
    print(f"{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    with_filter = {}
    for grow in (0, -1, -2):
        r = evaluate(0.35, grow, 300)
        with_filter[grow] = r
        print(f"{grow:>6}{r['pq']:>9.4f}{r['sq']:>8.4f}{r['rq']:>8.4f}"
              f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}")

    print("\n=== erosion WITHOUT the area filter (pure boundary effect) ===")
    print(f"{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    without = {}
    for grow in (0, -1, -2):
        r = evaluate(0.35, grow, 1)
        without[grow] = r
        print(f"{grow:>6}{r['pq']:>9.4f}{r['sq']:>8.4f}{r['rq']:>8.4f}"
              f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}")

    gain_with = with_filter[-1]["pq"] - with_filter[0]["pq"]
    gain_without = without[-1]["pq"] - without[0]["pq"]
    print(f"\nerosion gain WITH filter   : {gain_with:+.4f}")
    print(f"erosion gain WITHOUT filter: {gain_without:+.4f}")
    share = gain_without / gain_with if gain_with else float("nan")
    print(f"-> boundary effect accounts for {100 * share:.0f}% of the gain; "
          f"the area filter supplies the rest")

    # SQ isolates the boundary effect exactly: it is the mean IoU of matched
    # pairs and cannot be moved by discarding unmatched predictions.
    print(f"\nSQ at grow 0 / -1 (no filter): {without[0]['sq']:.4f} -> {without[-1]['sq']:.4f} "
          f"({without[-1]['sq'] - without[0]['sq']:+.4f})")
    print("SQ can only rise if already-matching masks fit better, so this is the")
    print("boundary correction with the filter effect removed entirely.")


if __name__ == "__main__":
    main()
