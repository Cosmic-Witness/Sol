"""Dihedral test-time augmentation with mask-level fusion.

Why this is worth running when the resolution sweep failed
----------------------------------------------------------
Inference above native resolution degraded badly: 2048 gives PQ 0.4064, 3072
gives 0.3624, and the mechanism was clear — segmentation quality stayed flat
while recognition collapsed, because inflating objects past the scales the
anchor-free head learned makes it stop proposing them.

The eight dihedral transforms do not touch scale. A flipped or rotated solar
disk is the same disk at the same size, and the annotation convention has no
canonical up direction, so all eight are physically valid views of identical
data. The negative result about scale says nothing about them, and treating one
failed augmentation as evidence against a different one would be an error.

Ultralytics refuses `augment=True` on segmentation models — it warns and
silently reverts to single-scale — so this is implemented directly.

Fusion
------
Averaging soft masks across views is not possible here because Ultralytics
returns binarised masks, and the instances are not aligned across views: view 3
may split a filament that view 5 keeps whole, and the ordering is arbitrary.

So masks are clustered first. The highest-scoring unclustered mask seeds a
cluster; every mask from any view overlapping it above `link_iou` joins. The
cluster's members are then averaged into a vote map and thresholded at
`vote_frac` of the number of views that contributed. That makes the correction
per-pixel — a pixel survives when the views agree about it — which is exactly
what a global erosion constant cannot do.

The vote threshold is a new knob and interacts with the erosion already in the
pipeline, since both trim boundaries. Both are swept together rather than
assumed independent.
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
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048

# (number of 90-degree rotations, whether to mirror first). The eight elements of
# the dihedral group of the square.
TRANSFORMS = [(k, m) for m in (False, True) for k in range(4)]


def apply_transform(image: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    if mirror:
        image = np.fliplr(image)
    return np.ascontiguousarray(np.rot90(image, k))


def invert_transform(mask: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    """Undo apply_transform. Rotation is inverted first, then the mirror."""
    mask = np.rot90(mask, -k)
    if mirror:
        mask = np.fliplr(mask)
    return np.ascontiguousarray(mask)


def predict_views(model, image_path: Path, imgsz: int, conf: float, max_det: int):
    """Run the model over all eight views; return masks in the original frame."""
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise SystemExit(f"cannot read {image_path}")

    out = []
    for k, mirror in TRANSFORMS:
        view = apply_transform(raw, k, mirror)
        result = model.predict(view, imgsz=imgsz, conf=conf, iou=0.60,
                               max_det=max_det, retina_masks=True, verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            continue
        data = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for index in range(len(data)):
            mask = data[index].astype(np.uint8)
            if mask.shape != (FULL, FULL):
                mask = cv2.resize(mask, (FULL, FULL), interpolation=cv2.INTER_NEAREST)
            restored = invert_transform(mask, k, mirror)
            if restored.sum() < 40:
                continue
            out.append((float(scores[index]),
                        mask_util.encode(np.asfortranarray(restored))))
    return out


def fuse(entries, link_iou: float, vote_frac: float, n_views: int):
    """Cluster masks across views and vote per pixel."""
    if not entries:
        return []
    order = sorted(range(len(entries)), key=lambda i: -entries[i][0])
    decoded = {i: mask_util.decode(entries[i][1]) for i in order}
    used = set()
    fused = []

    for seed in order:
        if seed in used:
            continue
        members = [seed]
        used.add(seed)
        seed_mask = decoded[seed]
        for other in order:
            if other in used:
                continue
            m = decoded[other]
            inter = int((seed_mask & m).sum())
            if not inter:
                continue
            union = int((seed_mask | m).sum())
            if inter / union >= link_iou:
                members.append(other)
                used.add(other)

        votes = np.zeros((FULL, FULL), np.uint8)
        for member in members:
            votes += decoded[member]
        # A pixel survives when enough of the views that saw this instance agree.
        needed = max(1, int(round(vote_frac * len(members))))
        binary = (votes >= needed).astype(np.uint8)
        if binary.sum() < 40:
            continue
        score = float(np.mean([entries[m][0] for m in members]))
        # Reward instances several views agree exists; a single-view detection is
        # more often a hallucination than a filament the others simply missed.
        support = len(members) / n_views
        fused.append((score * (0.5 + 0.5 * support), binary))
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--floor-conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--dump-cache", default=None)
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
    print(f"{len(records)} records over {len(photographs)} photographs, "
          f"{len(TRANSFORMS)} views each", flush=True)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_dir = Path(args.images)

    cache = {}
    for position, stem in enumerate(photographs, start=1):
        cache[stem] = predict_views(model, images_dir / stem, args.imgsz,
                                    args.floor_conf, args.max_det)
        if position % 10 == 0 or position == len(photographs):
            total = sum(len(v) for v in cache.values())
            print(f"  {position}/{len(photographs)} photographs, {total} view-masks", flush=True)

    if args.dump_cache:
        Path(args.dump_cache).write_text(json.dumps(
            {s: [(sc, {"size": r["size"], "counts": r["counts"].decode("ascii")})
                 for sc, r in v] for s, v in cache.items()}))
        print(f"wrote view cache to {args.dump_cache}", flush=True)

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

    truths = {i: gt_masks(a) for i, _s, a in records}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    results = []
    print(f"\n{'link':>6}{'vote':>7}{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    for link_iou in (0.4, 0.5):
        for vote_frac in (0.3, 0.5, 0.7):
            for grow in (0, -1):
                rows = []
                for image_id, stem, _ in records:
                    fused = fuse(cache.get(stem, []), link_iou, vote_frac, len(TRANSFORMS))
                    cands = []
                    for score, binary in fused:
                        m = cv2.erode(binary, kernel) if grow else binary
                        if int(m.sum()) >= 300:
                            cands.append((score, m))
                    painted = paint_panoptic(cands, min_area=300) if cands else []
                    rows.append(compute_pq([m for _, m, _ in painted], truths[image_id]))
                r = aggregate_pq(rows)
                r.update(link_iou=link_iou, vote_frac=vote_frac, grow=grow)
                results.append(r)
                print(f"{link_iou:>6.1f}{vote_frac:>7.1f}{grow:>6}{r['pq']:>9.4f}"
                      f"{r['sq']:>8.4f}{r['rq']:>8.4f}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}",
                      flush=True)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nBEST link={best['link_iou']} vote={best['vote_frac']} grow={best['grow']} "
          f"PQ={best['pq']:.4f}", flush=True)
    print("single-view baseline at this operating point: PQ 0.4404", flush=True)
    Path(args.out).write_text(json.dumps({"sweep": results, "best": best}, indent=2))


if __name__ == "__main__":
    main()
