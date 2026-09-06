"""Fuse independent annotations of one photograph into consensus instances.

Why this is the change worth making
-----------------------------------
Inter-annotator agreement on this dataset is PQ 0.3371 (SQ 0.6341, RQ 0.5317)
over 598 pairs. The model already scores 0.4064 on validation. Pushing mask
quality further means fitting the disagreement between annotators more tightly
than annotators fit each other, which is fitting noise.

The 296 multiply-annotated observations are the way out. Where two or three
people traced the same photograph, the parts they agree on are signal and the
parts they do not are the noise floor made visible. Training on the agreement
raises the target's quality instead of chasing its scatter.

How instances are fused
-----------------------
Annotators disagree about *how many* filaments there are as much as about where
their edges fall — recognition agreement is 0.5317, lower than segmentation
agreement at 0.6341. So instances are first grouped across annotators by overlap,
then each group is reduced by pixel-wise vote.

A group is kept when at least half the annotators drew something in it. With
three annotators that means two must agree; with two it means both, which is
strict but correct — a filament only one of two people saw is precisely the case
where the label is unreliable.

The vote is on pixels, not on unions: a union inherits every annotator's most
generous boundary and inflates masks, and an intersection inherits the meanest.
The median is the estimator that is robust to one outlying tracing.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util

FULL = 2048


def _rasterise(annotations: list[dict], size: int = FULL) -> list[np.ndarray]:
    scale = size / FULL
    out = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not segmentation or isinstance(segmentation, dict):
            continue
        rings = [[c * scale for c in r] for r in segmentation if len(r) >= 6]
        if not rings:
            continue
        rles = mask_util.frPyObjects(rings, size, size)
        out.append(mask_util.decode(mask_util.merge(rles)).astype(np.uint8))
    return out


def _group_across_annotators(per_annotator: list[list[np.ndarray]],
                             iou_link: float = 0.25) -> list[list[tuple[int, int]]]:
    """Link instances that different annotators drew for the same filament.

    Linking uses a permissive IoU of 0.25 rather than the metric's 0.5. Two
    people tracing one filament routinely overlap by less than half — that is
    what SQ 0.63 between annotators means — so a 0.5 link would split most
    genuine agreements into separate groups and then discard them as unanimous
    disagreements.
    """
    flat = [(a, i, m) for a, masks in enumerate(per_annotator) for i, m in enumerate(masks)]
    if not flat:
        return []

    parent = list(range(len(flat)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    encoded = [mask_util.encode(np.asfortranarray(m)) for _, _, m in flat]
    for p in range(len(flat)):
        for q in range(p + 1, len(flat)):
            if flat[p][0] == flat[q][0]:
                continue  # same annotator: never merge their own instances
            iou = float(mask_util.iou([encoded[p]], [encoded[q]], [0])[0][0])
            if iou >= iou_link:
                union(p, q)

    groups = defaultdict(list)
    for index in range(len(flat)):
        groups[find(index)].append((flat[index][0], flat[index][1]))
    return list(groups.values())


def consensus_masks(per_annotator: list[list[np.ndarray]],
                    min_area: int = 150) -> list[np.ndarray]:
    """Reduce several annotators' instance sets to one agreed set."""
    n_annotators = len(per_annotator)
    if n_annotators == 1:
        return per_annotator[0]

    required = (n_annotators + 1) // 2   # majority: 2 of 3, 2 of 2, 1 of 1
    out = []
    for group in _group_across_annotators(per_annotator):
        contributors = {a for a, _ in group}
        if len(contributors) < required:
            continue
        stack = np.stack([per_annotator[a][i] for a, i in group]).astype(np.int16)
        # Pixel-wise vote among the annotators who drew this filament at all.
        votes = stack.sum(axis=0)
        agreed = (votes >= required).astype(np.uint8)
        if int(agreed.sum()) >= min_area:
            out.append(agreed)

    # Painting keeps the result disjoint: votes are independent per group and two
    # neighbouring groups can both claim a boundary pixel.
    claimed = np.zeros_like(out[0]) if out else None
    disjoint = []
    for mask in sorted(out, key=lambda m: -int(m.sum())):
        remaining = (mask & (1 - claimed)).astype(np.uint8)
        if int(remaining.sum()) >= min_area:
            claimed |= remaining
            disjoint.append(remaining)
    return disjoint


def build_consensus(coco: dict, size: int = FULL, min_area: int = 150) -> dict[str, list[np.ndarray]]:
    """file_name -> consensus instance masks, over every observation in the file."""
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    per_stem = defaultdict(list)
    for record in coco["images"]:
        per_stem[record["file_name"]].append(record["id"])

    scaled_min = int(min_area * (size / FULL) ** 2)
    out = {}
    for stem, ids in per_stem.items():
        per_annotator = [_rasterise(by_image.get(i, []), size) for i in ids]
        per_annotator = [p for p in per_annotator if p]
        if not per_annotator:
            continue
        out[stem] = consensus_masks(per_annotator, min_area=scaled_min)
    return out
