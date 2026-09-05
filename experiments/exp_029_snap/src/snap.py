"""Snap mask boundaries to the image's own edges, predicting nothing.

SQ is 0.678 against a ceiling near 0.82, which is the largest lever left, and
every attempt to move it has been a model trained to predict the detector's
boundary error. exp_017 and exp_018 independently established that this error is
not predictable: a per-instance sub-pixel trim derived from the mask field does
no better than a global one-pixel erosion, and a U-Net given the crop and the
coarse mask learns the identity function.

This does not predict the error. A filament is a dark feature on a bright disk,
so its boundary is physically present in the pixels, and the mask can be pulled
onto it by looking at intensity rather than by inferring a correction. Two
classical mechanisms, both parameter-light and neither trained:

**Intensity snapping.** Within a narrow band around the current boundary, decide
each pixel by whether it is darker than a threshold taken from the neighbourhood
itself. Pixels far from the boundary are left alone, so the method can only move
the rim, which is where 64% of the disagreement lives.

**Guided filtering.** Smooth a softened mask under the guidance of the
photograph, so that mask gradients are encouraged to coincide with image
gradients, then re-threshold. This is the standard edge-aware refinement and it
respects the structure of the underlying image by construction.

The prototype basis that produces these masks is 32 smooth components at stride
4, which is band-limited by design; the photograph is not. That mismatch is the
reason to expect anything here at all.
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
BASE_CONF = 0.30
MIN_AREA = 250


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


def snap_intensity(image: np.ndarray, mask: np.ndarray, band: int,
                   bias: float) -> np.ndarray:
    """Re-decide the pixels within `band` of the boundary, by darkness.

    The threshold is the midpoint between what the mask currently calls filament
    and what it calls disk, taken locally from the instance itself rather than
    globally, since limb darkening makes any global cut wrong near the edge.
    `bias` shifts it: positive keeps more, negative keeps less.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 20:
        return mask
    pad = band + 2
    top, bottom = max(0, ys.min() - pad), min(FULL, ys.max() + pad + 1)
    left, right = max(0, xs.min() - pad), min(FULL, xs.max() + pad + 1)
    patch = image[top:bottom, left:right].astype(np.float32)
    local = mask[top:bottom, left:right]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    inner = cv2.erode(local, kernel)
    outer = cv2.dilate(local, kernel)
    ring = (outer > 0) & (inner == 0)          # the band that may change
    if not ring.any() or inner.sum() < 5:
        return mask

    surround = (outer == 0)
    if surround.sum() < 5:
        return mask
    dark = float(patch[inner > 0].mean())      # filament interior
    bright = float(patch[surround].mean())     # disk just outside
    if not np.isfinite(dark) or not np.isfinite(bright) or bright <= dark:
        return mask
    threshold = dark + (bright - dark) * (0.5 + bias)

    updated = local.copy()
    updated[ring] = (patch[ring] < threshold).astype(np.uint8)
    updated[inner > 0] = 1                     # never hollow out the interior

    out = mask.copy()
    out[top:bottom, left:right] = updated
    return out


def snap_guided(image: np.ndarray, mask: np.ndarray, radius: int,
                eps: float, level: float) -> np.ndarray:
    """Edge-aware smoothing of a softened mask, guided by the photograph."""
    ys, xs = np.nonzero(mask)
    if ys.size < 20:
        return mask
    pad = radius + 4
    top, bottom = max(0, ys.min() - pad), min(FULL, ys.max() + pad + 1)
    left, right = max(0, xs.min() - pad), min(FULL, xs.max() + pad + 1)
    guide = image[top:bottom, left:right].astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(mask[top:bottom, left:right].astype(np.float32),
                            (0, 0), sigmaX=1.5)

    try:
        filtered = cv2.ximgproc.guidedFilter(guide, soft, radius, eps)
    except AttributeError:
        # Without ximgproc, the same idea by its definition: a local linear
        # model of the soft mask on the guide, solved in closed form.
        mean_g = cv2.boxFilter(guide, -1, (radius, radius))
        mean_s = cv2.boxFilter(soft, -1, (radius, radius))
        corr_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))
        corr_gs = cv2.boxFilter(guide * soft, -1, (radius, radius))
        var_g = corr_gg - mean_g * mean_g
        cov_gs = corr_gs - mean_g * mean_s
        a = cov_gs / (var_g + eps)
        b = mean_s - a * mean_g
        filtered = (cv2.boxFilter(a, -1, (radius, radius)) * guide
                    + cv2.boxFilter(b, -1, (radius, radius)))

    out = mask.copy()
    out[top:bottom, left:right] = (filtered > level).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from pathlib import Path
    images_dir = Path(args.images)

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}
    split = make_split(args.annotations)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]
    truths = {iid: gt_masks(anns) for iid, _, anns in records}

    with open(args.candidates, encoding="utf-8") as fh:
        cached = json.load(fh)

    photographs = {}
    kept: dict[str, list[tuple[float, np.ndarray]]] = {}
    for stem, entries in cached.items():
        picked = [(e["features"]["score"], e["counts"]) for e in entries
                  if e["features"]["score"] >= BASE_CONF]
        if not picked:
            kept[stem] = []
            continue
        photographs[stem] = cv2.imread(str(images_dir / stem), cv2.IMREAD_GRAYSCALE)
        kept[stem] = [(s, decode_rle(c)) for s, c in picked]
    print(f"{sum(len(v) for v in kept.values())} candidates above {BASE_CONF}",
          flush=True)

    def evaluate(name: str, transform) -> dict:
        rows = []
        for iid, stem, _ in records:
            image = photographs.get(stem)
            candidates = []
            for score, mask in kept.get(stem, []):
                candidates.append((score, mask if transform is None or image is None
                                   else transform(image, mask)))
            painted = paint_panoptic(candidates, min_area=MIN_AREA) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], truths[iid]))
        row = aggregate_pq(rows)
        row["rule"] = name
        print(f"{name:<42} PQ {row['pq']:.4f}  SQ {row['sq']:.4f}  RQ {row['rq']:.4f}  "
              f"TP {row['tp']:4d} FP {row['fp']:4d} FN {row['fn']:4d}", flush=True)
        return row

    results = [evaluate("untouched (baseline)", None)]

    for band in (2, 3, 5):
        for bias in (-0.10, 0.0, 0.10):
            results.append(evaluate(
                f"intensity snap band {band} bias {bias:+.2f}",
                lambda im, m, b=band, x=bias: snap_intensity(im, m, b, x)))

    for radius in (4, 8):
        for level in (0.4, 0.5, 0.6):
            results.append(evaluate(
                f"guided filter r{radius} level {level}",
                lambda im, m, r=radius, l=level: snap_guided(im, m, r, 1e-3, l)))

    best = max(results, key=lambda r: r["pq"])
    baseline = results[0]
    print(f"\nbaseline {baseline['pq']:.4f} -> best {best['pq']:.4f} "
          f"({best['pq'] - baseline['pq']:+.4f})  [{best['rule']}]", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best, "baseline": baseline}, fh, indent=2)


if __name__ == "__main__":
    main()
