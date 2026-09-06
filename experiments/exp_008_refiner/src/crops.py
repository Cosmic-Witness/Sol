"""Build a boundary-refinement dataset from the annotations alone.

The target
----------
64% of the model's mask error lies within two pixels of the true boundary, and
eliminating it would take SQ from 0.679 to 0.855 — worth about +0.11 PQ. A
global one-pixel erosion recovered 0.023 of that, because one constant trim
cannot be right for every instance and one pixel is the quantisation floor.

What captures the rest is a model that predicts the boundary per pixel. That is
the second stage: crop a detection at native resolution, give a network the image
and the coarse mask, and have it return the exact mask.

Why the training pairs are synthetic
------------------------------------
The obvious way to build pairs is to run the detector over the training set and
match its outputs to ground truth. That costs an inference pass over 601
photographs at 2048, which is hours of CPU that buys nothing the annotations do
not already contain.

Instead the coarse input is *simulated* by degrading the ground truth in the ways
the detector's masks are actually wrong — measured, not guessed:

- dilation, because the masks run 11% fat from the rasterisation convention
- a small translation, because a low-rank prototype reconstruction drifts
- boundary noise, because the prototype basis is smooth and cannot follow a
  convoluted rim

The refiner never needs to have seen the real detector, only the *kind* of error
it makes, and this way every one of the 8199 annotations becomes a training pair
instead of only those the detector happened to find.

The target is rasterised with pycocotools — the scorer's own convention — so the
10.8% offset that Ultralytics' cv2.fillPoly introduces simply never enters this
stage.
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

FULL = 2048
CROP = 256


def degrade(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Imitate the detector's error modes on a ground-truth mask."""
    out = mask.copy()

    # Fat by default, occasionally thin: the measured bias is +11% area, but the
    # refiner should not learn that shrinking is always correct.
    grow = rng.choice([-1, 0, 1, 1, 2], p=[0.15, 0.15, 0.35, 0.20, 0.15])
    if grow:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(int(grow)) + 1,) * 2)
        out = (cv2.dilate if grow > 0 else cv2.erode)(out, kernel)

    # Sub-pixel drift, quantised to whole pixels by the raster.
    dx, dy = rng.integers(-2, 3, 2)
    if dx or dy:
        out = np.roll(np.roll(out, int(dy), axis=0), int(dx), axis=1)

    # Smooth the boundary: a 32-prototype basis cannot follow a convoluted rim,
    # so the coarse mask is a low-pass version of the truth.
    if rng.random() < 0.7:
        blurred = cv2.GaussianBlur(out.astype(np.float32), (0, 0), rng.uniform(1.0, 3.0))
        out = (blurred > rng.uniform(0.35, 0.55)).astype(np.uint8)

    return out.astype(np.uint8)


def build(annotations_path: str, images_dir: str, out_dir: str, per_instance: int) -> None:
    with open(annotations_path, encoding="utf-8") as fh:
        coco = json.load(fh)

    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    meta = {r["id"]: r for r in coco["images"]}

    split = make_split(annotations_path)
    assert_disjoint(split)
    fold_of = {i: "train" for i in split.train_image_ids}
    fold_of.update({i: "val" for i in split.val_image_ids})
    print(split.summary(), flush=True)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    source = Path(images_dir)

    buffers = {f: {"image": [], "coarse": [], "truth": []} for f in ("train", "val")}
    cached_stem, cached_image = None, None

    for image_id, record in meta.items():
        fold = fold_of.get(image_id)
        if fold is None:
            continue
        stem = record["file_name"]
        if stem != cached_stem:
            raw = cv2.imread(str(source / stem), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise SystemExit(f"cannot read {source / stem}")
            cached_image, cached_stem = raw, stem

        for annotation in by_image.get(image_id, []):
            seg = annotation.get("segmentation")
            if not seg or isinstance(seg, dict):
                continue
            rings = [r for r in seg if len(r) >= 6]
            if not rings:
                continue
            # pycocotools: the scorer's convention, so the offset never enters.
            truth = mask_util.decode(
                mask_util.merge(mask_util.frPyObjects(rings, FULL, FULL))).astype(np.uint8)
            if truth.sum() < 150:
                continue

            ys, xs = np.nonzero(truth)
            cy, cx = int(ys.mean()), int(xs.mean())

            for _ in range(per_instance if fold == "train" else 1):
                coarse = degrade(truth, rng)
                # Jitter the crop so the instance is not always centred; at
                # inference the crop is centred on a detection, which is itself
                # off-centre relative to the truth.
                oy, ox = rng.integers(-24, 25, 2)
                y0 = np.clip(cy + oy - CROP // 2, 0, FULL - CROP)
                x0 = np.clip(cx + ox - CROP // 2, 0, FULL - CROP)
                sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
                t = truth[sl]
                if t.sum() < 40:
                    continue
                buffers[fold]["image"].append(cached_image[sl])
                buffers[fold]["coarse"].append(coarse[sl])
                buffers[fold]["truth"].append(t)

    for fold, arrays in buffers.items():
        n = len(arrays["image"])
        for name in ("image", "coarse", "truth"):
            stacked = np.stack(arrays[name]).astype(np.uint8)
            np.save(root / f"{fold}_{name}.npy", stacked)
        print(f"{fold}: {n} crops of {CROP}x{CROP}", flush=True)
        if n:
            mean_iou = np.mean([
                (c & t).sum() / max((c | t).sum(), 1)
                for c, t in zip(arrays["coarse"][:500], arrays["truth"][:500])
            ])
            print(f"  coarse-vs-truth IoU {mean_iou:.4f} "
                  f"(the model's own SQ is 0.679, so the simulation is comparable)",
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-instance", type=int, default=3)
    args = parser.parse_args()
    build(args.annotations, args.images, args.output, args.per_instance)


if __name__ == "__main__":
    main()
