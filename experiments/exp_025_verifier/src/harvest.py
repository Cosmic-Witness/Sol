"""Crop every candidate the detector emits, and label whether it is a filament.

The question this feeds is the one exp_019 could not answer from fifteen
numbers: given a proposed mask, is there really a filament there? 83.2% of the
ground truth is covered by some candidate at IoU 0.5, but only 72% of those
clear the confidence floor, so 305 truths per validation pass are seen and
disbelieved. Deciding correctly which to believe is worth up to +0.09 PQ.

A candidate is positive when it matches a ground-truth instance above IoU 0.5 --
the same threshold the score uses, so the classifier is trained on exactly the
decision the metric rewards.

Crops are square, centred on the candidate, and sized to its extent with a
margin, then resampled to a fixed grid. Two channels go in: the photograph and
the proposed mask. Without the mask channel the model cannot tell which of
several nearby filaments it is being asked about.

Reads cached candidate RLEs, so there is no inference here and no GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util

from shared.data_split import make_split
from shared.utils import decode_rle

FULL = 2048
CROP = 128           # the classifier's input grid
MARGIN = 1.6         # crop side as a multiple of the candidate's longest extent
MIN_SIDE = 64        # never crop tighter than this in native pixels
MATCH = 0.5


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


def crop_of(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    side = max(MIN_SIDE, MARGIN * max(ys.max() - ys.min(), xs.max() - xs.min()))
    half = int(round(side / 2))
    top, left = int(round(cy)) - half, int(round(cx)) - half
    bottom, right = top + 2 * half, left + 2 * half

    # Pad rather than clamp, so a candidate near the limb keeps its centre in the
    # middle of the crop and the model never has to learn that off-centre means
    # "near the edge of the disk".
    pad = max(0, -top, -left, bottom - FULL, right - FULL)
    if pad:
        image = np.pad(image, pad, mode="constant")
        mask = np.pad(mask, pad, mode="constant")
        top += pad; left += pad; bottom += pad; right += pad
    patch = image[top:bottom, left:right]
    patch_mask = mask[top:bottom, left:right]
    if patch.size == 0:
        return None
    return (cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA),
            cv2.resize(patch_mask, (CROP, CROP), interpolation=cv2.INTER_NEAREST))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=None,
                        help="cached candidate RLEs; validation only")
    parser.add_argument("--weights", default=None,
                        help="run the detector instead, for photographs no cache covers")
    parser.add_argument("--floor-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=FULL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from pathlib import Path
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    ids_of_stem = defaultdict(list)
    for record in coco["images"]:
        ids_of_stem[record["file_name"]].append(record["id"])

    split = make_split(args.annotations)
    fold_of = {s: "train" for s in split.train_stems}
    fold_of.update({s: "val" for s in split.val_stems})

    # The cache exp_005 and exp_015 dumped covers the 106 validation
    # photographs only. Training crops are where the classifier learns, so those
    # candidates have to be generated -- on CPU, since GPU hours are spoken for
    # and this is a harvest rather than a submission.
    if args.candidates:
        with open(args.candidates, encoding="utf-8") as fh:
            cached = json.load(fh)
        model = None
    elif args.weights:
        cached = None
        from ultralytics import YOLO
        model = YOLO(args.weights)
    else:
        raise SystemExit("need --candidates or --weights")

    stems = sorted(cached) if cached is not None else sorted(
        p.name for p in images_dir.glob("*.jpeg") if p.name in fold_of)
    if args.limit:
        stems = stems[:args.limit]

    def candidates_for(stem: str) -> tuple[list[np.ndarray], list[float]]:
        if cached is not None:
            entries = cached[stem]
            return ([decode_rle(e["counts"]) for e in entries],
                    [e["features"]["score"] for e in entries])
        result = model.predict(str(images_dir / stem), imgsz=args.imgsz,
                               conf=args.floor_conf, iou=0.60, max_det=100,
                               retina_masks=True, verbose=False)[0]
        if result.masks is None or not len(result.masks.data):
            return [], []
        raw = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        out = []
        for index in range(raw.shape[0]):
            m = raw[index].astype(np.uint8)
            if m.shape != (FULL, FULL):
                m = (cv2.resize(m.astype(np.float32), (FULL, FULL),
                                interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
            out.append(m)
        return out, [float(c) for c in confs]

    buckets: dict[str, dict[str, list]] = {
        "train": {"image": [], "mask": [], "label": [], "score": []},
        "val": {"image": [], "mask": [], "label": [], "score": []},
    }

    for position, stem in enumerate(stems, start=1):
        fold = fold_of.get(stem)
        if fold is None:
            continue
        photograph = cv2.imread(str(images_dir / stem), cv2.IMREAD_GRAYSCALE)
        if photograph is None:
            continue

        # A candidate counts as real if ANY annotator drew a filament there. Two
        # people disagree about a fifth of instances, so labelling against one of
        # them would teach the classifier that a filament the other one drew is
        # a false positive.
        truths = [m for rid in ids_of_stem.get(stem, []) for m in gt_masks(by_image.get(rid, []))]

        masks, scores = candidates_for(stem)
        if not masks:
            continue
        if truths:
            enc_p = mask_util.encode(np.asfortranarray(np.stack(masks, axis=-1)))
            enc_t = mask_util.encode(np.asfortranarray(np.stack(truths, axis=-1)))
            best = mask_util.iou(enc_p, enc_t, [0] * len(truths)).max(axis=1)
        else:
            best = np.zeros(len(masks))

        for index, mask in enumerate(masks):
            cropped = crop_of(photograph, mask)
            if cropped is None:
                continue
            patch, patch_mask = cropped
            buckets[fold]["image"].append(patch)
            buckets[fold]["mask"].append(patch_mask * 255)
            buckets[fold]["label"].append(int(best[index] >= MATCH))
            buckets[fold]["score"].append(float(scores[index]))

        if position % 25 == 0 or position == len(stems):
            total = sum(len(v["label"]) for v in buckets.values())
            print(f"  {position}/{len(stems)} photographs, {total} crops", flush=True)

    for fold, data in buckets.items():
        if not data["label"]:
            continue
        labels = np.array(data["label"], dtype=np.uint8)
        np.save(out_dir / f"{fold}_image.npy", np.stack(data["image"]))
        np.save(out_dir / f"{fold}_mask.npy", np.stack(data["mask"]))
        np.save(out_dir / f"{fold}_label.npy", labels)
        np.save(out_dir / f"{fold}_score.npy", np.array(data["score"], dtype=np.float32))
        print(f"{fold}: {len(labels)} crops, {labels.mean():.2%} positive", flush=True)


if __name__ == "__main__":
    main()
