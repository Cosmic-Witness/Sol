"""Choose the confidence and minimum-area thresholds on the validation fold.

Why this is worth a separate step
---------------------------------
Panoptic Quality is
    PQ = sum(IoU over matched pairs) / (TP + 0.5*FP + 0.5*FN)
so a false positive and a false negative cost the same half-unit. The confidence
threshold is the single knob that trades one directly against the other, and its
best value is a property of the trained model, not a constant worth guessing.
The public 0.55 notebook hard-codes 0.3 with no stated justification.

Cost control
------------
Inference runs **once** per photograph at a permissively low confidence, and the
resulting candidates are cached. Every threshold in the sweep is then evaluated
by filtering that cache rather than by re-running the model, which turns an
O(thresholds x images) GPU cost into O(images).

Masks are cached RLE-encoded rather than as arrays: a hundred 2048x2048 uint8
masks is 400 MB held at once, while their RLEs are a few kilobytes.

Note on the validation fold
---------------------------
180 validation records cover only 106 distinct photographs, because some were
annotated by two or three people. Inference is keyed by photograph and reused
across that photograph's records, so the model is not run three times on one
image. Each record is scored against its own annotator's ground truth, which is
the honest thing to do: annotators genuinely disagree, and averaging over that
disagreement is what the leaderboard does too.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

from shared.data_split import assert_disjoint, make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048


def gt_masks_for(annotations: list[dict]) -> list[np.ndarray]:
    """Rasterise one record's polygon annotations at full resolution."""
    masks = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not segmentation or isinstance(segmentation, dict):
            continue
        rles = mask_util.frPyObjects(segmentation, FULL, FULL)
        merged = mask_util.merge(rles) if isinstance(rles, list) else rles
        masks.append(mask_util.decode(merged).astype(np.uint8))
    return masks


def cache_candidates(model, image_path: Path, imgsz: int, floor_conf: float,
                     max_det: int, tta: bool) -> list[tuple[float, dict]]:
    """Run the model once; return (confidence, RLE) for every candidate."""
    import cv2

    result = model.predict(
        str(image_path), imgsz=imgsz, conf=floor_conf, iou=0.60,
        max_det=max_det, augment=tta, retina_masks=True, verbose=False,
    )[0]
    if result.masks is None or len(result.masks.data) == 0:
        return []

    raw = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    cached = []
    for index in range(len(raw)):
        mask = raw[index].astype(np.float32)
        if mask.shape != (FULL, FULL):
            mask = cv2.resize(mask, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
        binary = np.asfortranarray((mask > 0.5).astype(np.uint8))
        cached.append((float(scores[index]), mask_util.encode(binary)))
    return cached


def evaluate(cache: dict[str, list], records: list[tuple[str, str, list]],
             conf: float, min_area: int) -> dict:
    """Score one (conf, min_area) setting over every validation record."""
    per_image = []
    for _image_id, stem, annotations in records:
        candidates = [
            (score, mask_util.decode(rle))
            for score, rle in cache.get(stem, [])
            if score >= conf
        ]
        painted = paint_panoptic(candidates, min_area=min_area) if candidates else []
        per_image.append(compute_pq([m for _, m, _ in painted], gt_masks_for(annotations)))
    return aggregate_pq(per_image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--floor-conf", type=float, default=0.05,
                        help="inference threshold; every swept value must exceed it")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", default=None, help="write the sweep table here as JSON")
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
    photographs = sorted({stem for _, stem, _ in records})
    print(f"{len(records)} validation records over {len(photographs)} photographs", flush=True)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_dir = Path(args.images)

    cache: dict[str, list] = {}
    for position, stem in enumerate(photographs, start=1):
        cache[stem] = cache_candidates(
            model, images_dir / stem, args.imgsz, args.floor_conf, args.max_det, args.tta
        )
        if position % 20 == 0 or position == len(photographs):
            print(f"  inferred {position}/{len(photographs)}", flush=True)

    total = sum(len(v) for v in cache.values())
    print(f"cached {total} candidates above conf {args.floor_conf}\n", flush=True)

    results = []
    print(f"{'conf':>6} {'min_area':>9} {'PQ':>8} {'SQ':>7} {'RQ':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    for conf in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        for min_area in (100, 150, 300):
            r = evaluate(cache, records, conf, min_area)
            r.update(conf=conf, min_area=min_area)
            results.append(r)
            print(f"{conf:>6.2f} {min_area:>9d} {r['pq']:>8.4f} {r['sq']:>7.4f} "
                  f"{r['rq']:>7.4f} {r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nBEST  conf={best['conf']} min_area={best['min_area']} "
          f"PQ={best['pq']:.4f} (SQ {best['sq']:.4f} x RQ {best['rq']:.4f})", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({"sweep": results, "best": best}, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
