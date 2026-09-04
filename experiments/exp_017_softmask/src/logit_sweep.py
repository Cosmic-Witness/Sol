"""Sub-pixel mask trimming, which the erosion analysis wrongly declared impossible.

The claim was: the model's masks are systematically about half a pixel fat, the
correction has to be sub-pixel, and "on a raster the finest available operation
is one whole pixel -- the distance transform is quantised, so no sub-pixel
erosion exists". True of morphology on a binary mask. Not true of the mask
before it is binarised.

Ultralytics' `process_mask_native` builds a continuous field from the prototype
basis and cuts it at logit zero:

    scale_masks(coeffs.view(-1, mh, mw)[None], shape)[0].gt_(0.0)

That threshold is a free parameter. Raising it moves the boundary inward by a
distance set by the local gradient of the field -- continuous, not quantised, and
different for each instance rather than the same whole pixel everywhere. It is
exactly the instrument the one-pixel erosion was a blunt substitute for, and it
recovered only a fifth of what the rim is worth.

Inference runs once. The soft field is thresholded at every candidate level in
the same pass and each result cached as RLE, so the sweep over confidence, area
floor, erosion and logit level costs no further forward passes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

from shared.data_split import make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048

# Zero is the shipped cut. Positive trims inward, negative grows outward. A
# smoke test on one detection put the field's response at roughly 27% of the
# area per 0.25 of logit, so the grid is dense near zero: a half-pixel trim on a
# filament is a single-digit percentage of its area, and 0.25 would overshoot it
# as badly as the one-pixel erosion does.
LOGIT_LEVELS = [-0.4, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2,
                0.3, 0.4, 0.6, 0.9, 1.3, 2.0]


# Filled by the patched `process_mask_native`, one entry per predict call: a list
# of per-instance lists of RLEs, indexed by logit level.
STASH: list = []


def patch_ultralytics() -> None:
    """Make `process_mask_native` record the field's other cuts on its way past.

    The returned tensor keeps its shape and its meaning, so nothing downstream of
    the predictor changes; the alternative thresholds are encoded to RLE inside
    the function and left in `STASH`. Doing the encoding here rather than
    returning a taller tensor also keeps the upstream memory discipline intact --
    the chunked upsampling exists to avoid an N x H x W float intermediate, and
    each chunk is reduced to bytes before the next is built.
    """
    from ultralytics.utils import ops

    def process_mask_recording(protos, masks_in, bboxes, shape):
        c, mh, mw = protos.shape
        h, w = shape
        if masks_in.shape[0] == 0:
            STASH.append([])
            return torch.zeros((0, h, w), dtype=torch.uint8, device=masks_in.device)
        coeffs = masks_in @ protos.float().view(c, -1)
        step = max(1, 32_000_000 // (h * w))
        per_level: list[list] = [[] for _ in LOGIT_LEVELS]
        for i in range(0, coeffs.shape[0], step):
            field = ops.scale_masks(coeffs[i:i + step].view(-1, mh, mw)[None], shape)[0]
            for index, level in enumerate(LOGIT_LEVELS):
                cut = ops.crop_mask(field.gt(level).byte(), bboxes[i:i + step])
                per_level[index].append(cut.cpu().numpy())
            del field

        planes = [np.concatenate(chunks) for chunks in per_level]   # levels x (N,H,W)
        n = planes[0].shape[0]
        STASH.append([
            [mask_util.encode(np.asfortranarray(planes[level][j]))
             for level in range(len(LOGIT_LEVELS))]
            for j in range(n)
        ])
        # Level 0.0 is the shipped cut; hand that back so the caller sees exactly
        # what it would have seen unpatched.
        shipped = LOGIT_LEVELS.index(0.0)
        return torch.from_numpy(planes[shipped]).to(masks_in.device)

    ops.process_mask_native = process_mask_recording
    import ultralytics.models.yolo.segment.predict as segment_predict
    if hasattr(segment_predict, "process_mask_native"):
        segment_predict.process_mask_native = process_mask_recording


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


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if not pixels:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(mask, kernel).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--imgsz", type=int, default=FULL)
    parser.add_argument("--floor-conf", type=float, default=0.10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump-cache", default=None)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}
    split = make_split(args.annotations)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]
    photographs = sorted({stem for _, stem, _ in records})

    patch_ultralytics()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    images_dir = Path(args.images)

    # cache[stem] = list of (score, [rle per logit level])
    cache: dict[str, list] = {}
    for position, stem in enumerate(photographs, start=1):
        STASH.clear()
        result = model.predict(
            source=str(images_dir / stem), imgsz=args.imgsz,
            conf=args.floor_conf, iou=0.7, max_det=300,
            retina_masks=True, verbose=False,
        )[0]
        scores = result.boxes.conf.cpu().numpy()
        recorded = STASH[-1] if STASH else []
        if len(recorded) != len(scores):
            raise SystemExit(
                f"{stem}: recorded {len(recorded)} masks for {len(scores)} boxes; "
                "the patch is not seeing the same call the predictor uses")
        cache[stem] = [{"score": float(scores[j]), "rle": recorded[j]}
                       for j in range(len(scores))]
        if position % 20 == 0 or position == len(photographs):
            print(f"  inferred {position}/{len(photographs)}", flush=True)

    print(f"cached {sum(len(v) for v in cache.values())} candidates "
          f"at {len(LOGIT_LEVELS)} levels\n", flush=True)

    if args.dump_cache:
        serialisable = {
            stem: [{"score": e["score"],
                    "counts": [r["counts"].decode("ascii") for r in e["rle"]]}
                   for e in entries]
            for stem, entries in cache.items()
        }
        Path(args.dump_cache).write_text(json.dumps(serialisable))

    def evaluate(level_index: int, conf: float, min_area: int, grow: int) -> dict:
        rows = []
        for _iid, stem, annotations in records:
            candidates = []
            for entry in cache.get(stem, []):
                if entry["score"] < conf:
                    continue
                mask = mask_util.decode(entry["rle"][level_index])
                candidates.append((entry["score"], dilate(mask, grow)))
            painted = paint_panoptic(candidates, min_area=min_area) if candidates else []
            rows.append(compute_pq([m for _, m, _ in painted], gt_masks(annotations)))
        return aggregate_pq(rows)

    results = []
    print(f"{'logit':>7}{'conf':>6}{'area':>6}{'grow':>6}"
          f"{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    # The one-pixel erosion and a positive logit level are two spellings of the
    # same correction, so the grid crosses them: if the sub-pixel cut is the
    # better instrument, the winner sits at grow 0 and a positive level.
    for level_index, level in enumerate(LOGIT_LEVELS):
        for conf in (0.30, 0.35, 0.40):
            for grow in (0, -1):
                row = evaluate(level_index, conf, 300, grow)
                row.update({"logit": level, "conf": conf,
                            "min_area": 300, "grow": grow})
                results.append(row)
                print(f"{level:7.2f}{conf:6.2f}{300:6d}{grow:6d}"
                      f"{row['pq']:9.4f}{row['sq']:8.4f}{row['rq']:8.4f}"
                      f"{row['tp']:7d}{row['fp']:7d}{row['fn']:7d}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nbest: {json.dumps(best, indent=2)}")
    Path(args.out).write_text(json.dumps({"sweep": results, "best": best}, indent=2))


if __name__ == "__main__":
    main()
