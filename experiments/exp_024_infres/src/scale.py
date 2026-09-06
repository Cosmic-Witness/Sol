"""Does the 2048-trained model peak above 2048 at inference?

The project's resolution finding was measured on exp_002, which trained at 1280:
inference at 2048 beat 1280 by 0.033 PQ, and 2560 and 3072 degraded sharply. The
reading given at the time was that the anchor-free head responds over a limited
range of object sizes, and that the range is fixed at training time.

If that reading is right it does not transfer, because exp_010 trained at 2048.
exp_002 peaked at 1.6 times its training resolution; the same ratio would put
exp_010's peak near 3072. The earlier table is evidence about a 1280-trained
model and says nothing about this one.

Inference only -- no training, no new targets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from shared.data_split import make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048


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


def morph(binary: np.ndarray, pixels: int) -> np.ndarray:
    if not pixels:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(binary, kernel).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", default="1792,2048,2304,2560,3072")
    parser.add_argument("--floor-conf", type=float, default=0.15)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    stem_of = {r["id"]: r["file_name"] for r in coco["images"]}
    split = make_split(args.annotations)
    records = [(i, stem_of[i], by_image.get(i, [])) for i in split.val_image_ids]
    photographs = sorted({stem for _, stem, _ in records})
    truths = {iid: gt_masks(anns) for iid, _, anns in records}

    from ultralytics import YOLO
    model = YOLO(args.weights)
    images_dir = Path(args.images)

    results = []
    print(f"{'imgsz':>7}{'conf':>6}{'grow':>6}{'PQ':>9}{'SQ':>8}{'RQ':>8}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}", flush=True)

    for size in sizes:
        # Masks always come back at the native 2048 frame via retina_masks, so
        # every size is scored against the same ground truth.
        cache: dict[str, list[tuple[float, np.ndarray]]] = {}
        for stem in photographs:
            result = model.predict(str(images_dir / stem), imgsz=size,
                                   conf=args.floor_conf, iou=0.60, max_det=100,
                                   retina_masks=True, verbose=False)[0]
            entries = []
            if result.masks is not None and len(result.masks.data):
                raw = result.masks.data.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for index in range(raw.shape[0]):
                    mask = raw[index].astype(np.uint8)
                    if mask.shape != (FULL, FULL):
                        mask = (cv2.resize(mask.astype(np.float32), (FULL, FULL),
                                           interpolation=cv2.INTER_LINEAR) > 0.5
                                ).astype(np.uint8)
                    entries.append((float(scores[index]), mask))
            cache[stem] = entries

        for grow in (0, 1):
            grown = {s: [(sc, morph(m, grow)) for sc, m in v] for s, v in cache.items()}
            for conf in (0.20, 0.25, 0.30, 0.35):
                rows = []
                for iid, stem, _ in records:
                    picked = [(sc, m) for sc, m in grown.get(stem, []) if sc >= conf]
                    painted = paint_panoptic(picked, min_area=300) if picked else []
                    rows.append(compute_pq([m for _, m, _ in painted], truths[iid]))
                row = aggregate_pq(rows)
                row.update({"imgsz": size, "conf": conf, "grow": grow})
                results.append(row)
                print(f"{size:7d}{conf:6.2f}{grow:6d}{row['pq']:9.4f}"
                      f"{row['sq']:8.4f}{row['rq']:8.4f}"
                      f"{row['tp']:6d}{row['fp']:6d}{row['fn']:6d}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    print(f"\nbest: {json.dumps(best, indent=2)}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sweep": results, "best": best}, fh, indent=2)


if __name__ == "__main__":
    main()
