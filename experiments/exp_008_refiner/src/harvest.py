"""Harvest real detector-versus-truth pairs for refiner training.

Why this replaces the synthetic pairs
-------------------------------------
The first refiner was trained on ground truth that had been dilated, translated
and smoothed to imitate the detector's mistakes. It reached IoU 0.8529 on that
task and then lost to a parameterless 1px erosion on real output, 0.4322 against
0.4404.

The simulation was calibrated on severity — coarse IoU 0.72 against the
detector's real SQ of 0.679 — and that was mistaken for calibration on
character. Real errors are correlated with image content: faint filaments near
the limb, neighbours competing for the same 32 mask prototypes, boundaries that
drift where contrast falls. No hand-specified corruption reproduces that joint
structure.

So the coarse masks here are the detector's actual predictions, matched to the
ground truth they overlap most. The refiner then learns the error that exists
rather than the error that was imagined.

Matching
--------
A prediction is paired with the ground-truth instance it overlaps most, and kept
only if that overlap is at least `min_iou`. Below that the prediction is not a
damaged version of the truth, it is a different object or a hallucination, and
training a boundary refiner to "fix" it would teach it to relocate masks rather
than adjust them.

Predictions with no ground-truth overlap are recorded separately: they are what a
rejection head would need, and they are the population a boundary refiner
explicitly cannot help with.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

from experiments.exp_002_yolo_seg.src.predict import masks_for_image
from shared.data_split import assert_disjoint, make_split

FULL = 2048
CROP = 256


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
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="lower than the operating point: near-misses are exactly "
                             "the population the refiner must learn to rescue")
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="0 = all photographs")
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    meta = {r["id"]: r for r in coco["images"]}

    split = make_split(args.annotations)
    assert_disjoint(split)
    fold_of = {i: "train" for i in split.train_image_ids}
    fold_of.update({i: "val" for i in split.val_image_ids})

    # One photograph carries several annotation records; the detector's output
    # does not depend on which annotator's labels we are comparing to, so run it
    # once per photograph and pair against each record's truth.
    by_stem = defaultdict(list)
    for image_id, record in meta.items():
        if image_id in fold_of:
            by_stem[record["file_name"]].append(image_id)
    stems = sorted(by_stem)
    if args.limit:
        stems = stems[:args.limit]
    print(f"{len(stems)} photographs", flush=True)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_dir = Path(args.images)
    buffers = {f: {"image": [], "coarse": [], "truth": []} for f in ("train", "val")}
    unmatched = {"train": 0, "val": 0}
    ious = []

    for position, stem in enumerate(stems, start=1):
        raw = cv2.imread(str(images_dir / stem), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        candidates = masks_for_image(model, images_dir / stem, args.imgsz,
                                     args.conf, 0.60, 100, 150, False, 0)
        if not candidates:
            continue

        for image_id in by_stem[stem]:
            fold = fold_of[image_id]
            truths = gt_masks(by_image.get(image_id, []))
            if not truths:
                continue
            for _score, coarse in candidates:
                best_iou, best = 0.0, None
                for truth in truths:
                    inter = int((coarse & truth).sum())
                    if not inter:
                        continue
                    union = int((coarse | truth).sum())
                    value = inter / union
                    if value > best_iou:
                        best_iou, best = value, truth
                if best is None or best_iou < args.min_iou:
                    unmatched[fold] += 1
                    continue

                ious.append(best_iou)
                ys, xs = np.nonzero(best)
                cy, cx = int(ys.mean()), int(xs.mean())
                y0 = int(np.clip(cy - CROP // 2, 0, FULL - CROP))
                x0 = int(np.clip(cx - CROP // 2, 0, FULL - CROP))
                sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
                if best[sl].sum() < 40:
                    continue
                buffers[fold]["image"].append(raw[sl])
                buffers[fold]["coarse"].append(coarse[sl])
                buffers[fold]["truth"].append(best[sl])

        if position % 20 == 0 or position == len(stems):
            total = sum(len(b["image"]) for b in buffers.values())
            print(f"  {position}/{len(stems)} photographs, {total} pairs", flush=True)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for fold, arrays in buffers.items():
        n = len(arrays["image"])
        if n == 0:
            print(f"{fold}: no pairs", flush=True)
            continue
        for name in ("image", "coarse", "truth"):
            np.save(root / f"{fold}_{name}.npy", np.stack(arrays[name]).astype(np.uint8))
        print(f"{fold}: {n} real pairs", flush=True)

    if ious:
        print(f"\ncoarse-vs-truth IoU over matched pairs: mean {np.mean(ious):.4f}, "
              f"median {np.median(ious):.4f}", flush=True)
        print(f"unmatched predictions (below IoU {args.min_iou}): "
              f"train {unmatched['train']}, val {unmatched['val']}", flush=True)
        print("  those are not damaged truths and are excluded: teaching a boundary "
              "refiner to fix them would teach it to relocate masks", flush=True)


if __name__ == "__main__":
    main()
