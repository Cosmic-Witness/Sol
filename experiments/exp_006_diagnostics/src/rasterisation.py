"""Do the three common polygon rasterisers agree, and by how much on filaments?

Why this matters more here than anywhere else
---------------------------------------------
Post-hoc erosion of exactly one pixel improved the leaderboard 0.33 to 0.36, and
the improvement is close to uniform across 2531 candidates on 106 photographs.
A bias caused by coarse loss supervision ought to vary — worse on thin faint
filaments, milder on thick bright ones. A bias that is the same everywhere looks
like a *constant*, and a rasterisation convention is a constant.

Turning a polygon into a binary mask requires deciding which boundary pixels are
inside. pycocotools, cv2.fillPoly and PIL.ImageDraw each answer differently. For
a compact blob a one-pixel disagreement is a rounding error. A filament is nearly
all perimeter — long, thin, and highly convoluted — so the same disagreement is a
large fraction of its area.

If the training targets were rasterised with a more generous convention than the
scorer uses, the model is faithfully reproducing masks that are fat *by
construction*. No amount of finer loss supervision would fix that, because the
targets themselves carry the offset. The fix would be one line in target
generation, and the planned retrain would be wasted money.

The perimeter-to-area ratio is reported alongside, because it predicts how much
any convention difference costs: the disagreement lives on the boundary, so its
cost scales with how much boundary an object has per unit of area.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw

FULL = 2048


def raster_pycocotools(rings: list[list[float]], size: int) -> np.ndarray:
    rles = mask_util.frPyObjects(rings, size, size)
    return mask_util.decode(mask_util.merge(rles)).astype(np.uint8)


def raster_cv2(rings: list[list[float]], size: int) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    polys = [np.asarray(r, np.float32).reshape(-1, 2).round().astype(np.int32) for r in rings]
    cv2.fillPoly(mask, polys, 1)
    return mask


def raster_pil(rings: list[list[float]], size: int) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for ring in rings:
        draw.polygon([tuple(p) for p in np.asarray(ring, np.float32).reshape(-1, 2)], fill=1)
    return np.asarray(image, dtype=np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    return intersection / union if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--limit", type=int, default=400, help="instances to sample")
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)

    rng = np.random.default_rng(2026)
    annotations = [a for a in coco["annotations"]
                   if isinstance(a.get("segmentation"), list) and a["segmentation"]]
    sample = rng.choice(len(annotations), size=min(args.limit, len(annotations)), replace=False)

    pairs = {"coco_vs_cv2": [], "coco_vs_pil": [], "cv2_vs_pil": []}
    areas = {"coco": [], "cv2": [], "pil": []}
    shape_stats = []

    for index in sample:
        rings = [r for r in annotations[index]["segmentation"] if len(r) >= 6]
        if not rings:
            continue
        a = raster_pycocotools(rings, FULL)
        b = raster_cv2(rings, FULL)
        c = raster_pil(rings, FULL)
        if a.sum() == 0 or b.sum() == 0 or c.sum() == 0:
            continue

        pairs["coco_vs_cv2"].append(iou(a, b))
        pairs["coco_vs_pil"].append(iou(a, c))
        pairs["cv2_vs_pil"].append(iou(b, c))
        areas["coco"].append(int(a.sum()))
        areas["cv2"].append(int(b.sum()))
        areas["pil"].append(int(c.sum()))

        contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(len(c_) for c_ in contours)
        shape_stats.append(perimeter / max(int(a.sum()), 1))

    n = len(pairs["coco_vs_cv2"])
    print(f"instances compared: {n}\n")

    print("pairwise IoU between rasterisers:")
    for name, values in pairs.items():
        v = np.asarray(values)
        print(f"  {name:14s} mean {v.mean():.4f}  median {np.median(v):.4f}  "
              f"p05 {np.percentile(v, 5):.4f}  min {v.min():.4f}")

    print("\nmean instance area (px):")
    base = np.mean(areas["coco"])
    for name, values in areas.items():
        m = np.mean(values)
        print(f"  {name:14s} {m:9.1f}   ({100 * (m / base - 1):+.2f}% vs pycocotools)")

    perimeter_ratio = np.asarray(shape_stats)
    print(f"\nperimeter/area ratio: mean {perimeter_ratio.mean():.4f}, "
          f"median {np.median(perimeter_ratio):.4f}")
    print("  (a disagreement lives on the boundary, so its cost scales with this)")

    # A one-pixel erosion, for scale: how much area does the observed gain remove?
    eroded_iou = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for index in sample[:120]:
        rings = [r for r in annotations[index]["segmentation"] if len(r) >= 6]
        if not rings:
            continue
        a = raster_pycocotools(rings, FULL)
        if a.sum() == 0:
            continue
        eroded_iou.append(iou(a, cv2.erode(a, kernel)))
    print(f"\nIoU(mask, mask eroded 1px): mean {np.mean(eroded_iou):.4f}")
    print("  1px of erosion costs this much overlap on a real filament — the scale")
    print("  any rasterisation disagreement must be compared against.")


if __name__ == "__main__":
    main()
