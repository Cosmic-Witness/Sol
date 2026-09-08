"""Frozen exp_002 inference experiments, canonical pooled PQ and compact caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from shared.data_split import make_split, assert_disjoint
from shared.utils import aggregate_pq, paint_panoptic

BASELINE_PQ = 0.4403668270817509
VARIANTS = ["identity", "gamma_0.8", "gamma_1.2", "gain_0.85", "gain_1.15",
            "contrast_0.8", "contrast_1.2", "shift_1", "shift_2", "shift_-2"]


def transform(image, variant):
    if variant == "identity":
        return image
    operation, value = variant.rsplit("_", 1)
    value = float(value)
    if operation == "shift":
        return cv2.warpAffine(image, np.float32([[1, 0, value], [0, 1, value]]),
                              (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
    if operation == "gamma":
        table = np.clip(255 * (np.arange(256) / 255.)**value, 0, 255).astype(np.uint8)
        return cv2.LUT(image, table)
    if operation == "gain":
        return np.clip(image.astype(np.float32) * value, 0, 255).astype(np.uint8)
    if operation == "contrast":
        # Anchor contrast to the disk intensity, not to the black background.
        quiet = float(np.median(image[image > 40]))
        return np.clip(quiet + (image.astype(np.float32) - quiet) * value, 0, 255).astype(np.uint8)
    raise ValueError(variant)


def inverse_mask(mask, variant):
    if variant.startswith("shift_"):
        shift = -int(variant.rsplit("_", 1)[1])
        return cv2.warpAffine(mask, np.float32([[1, 0, shift], [0, 1, shift]]),
                              (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    return mask


def encode(mask):
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def pq_rle(predicted, truth):
    """Same greedy strict->0.5 matching as shared.compute_pq, without decoding."""
    matched_p, matched_g, total = set(), set(), 0.
    if predicted and truth:
        ious = np.asarray(mu.iou(predicted, truth, [0] * len(truth)))
        pairs = np.argwhere(ious > .5)
        for p, g in sorted(pairs, key=lambda pair: -ious[pair[0], pair[1]]):
            if p not in matched_p and g not in matched_g:
                matched_p.add(int(p)); matched_g.add(int(g)); total += float(ious[p, g])
    tp = len(matched_p)
    return dict(tp=tp, fp=len(predicted)-tp, fn=len(truth)-tp, iou_sum=total)


def records_for(annotations):
    coco = json.loads(Path(annotations).read_text())
    split = make_split(str(annotations)); assert_disjoint(split)
    record = {r["id"]: r for r in coco["images"]}
    by_id = defaultdict(list)
    val_ids = set(split.val_image_ids)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in val_ids:
            continue
        r = record[annotation["image_id"]]
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mu.merge(mu.frPyObjects(segmentation, r["height"], r["width"]))
        elif isinstance(segmentation["counts"], list):
            rle = mu.frPyObjects(segmentation, r["height"], r["width"])
        else:
            rle = segmentation
        by_id[annotation["image_id"]].append(rle)
    by_photo = defaultdict(list)
    for image_id in split.val_image_ids:
        by_photo[record[image_id]["file_name"]].append((image_id, by_id[image_id]))
    return dict(by_photo), split


def evaluate(cache, records, conf=.35, grow=-1, min_area=300):
    rows = []
    per_record = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for name, passes in records.items():
        candidates = []
        for entry in cache[name]:
            if entry["score"] < conf:
                continue
            mask = mu.decode(entry["rle"])
            if grow < 0:
                mask = cv2.erode(mask, kernel, iterations=-grow)
            elif grow != 0:
                raise ValueError("Only unaltered and eroded masks are screened")
            candidates.append((entry["score"], mask))
        predicted = [encode(m) for _, m, _ in paint_panoptic(candidates, min_area=min_area)]
        for image_id, truth in passes:
            score = pq_rle(predicted, truth)
            rows.append(score)
            per_record[image_id] = dict(score, photograph=name)
    return dict(aggregate_pq(rows), conf=conf, grow=grow, min_area=min_area), per_record


def evaluate_thresholds(cache, records, thresholds, grow=-1, min_area=300):
    """Paint once: every higher confidence threshold selects an identical prefix."""
    thresholds = sorted(set(thresholds))
    totals = {c: [] for c in thresholds}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for name, passes in records.items():
        candidates = []
        for entry in cache[name]:
            if entry['score'] < thresholds[0]:
                continue
            mask = mu.decode(entry['rle'])
            if grow < 0:
                mask = cv2.erode(mask, kernel, iterations=-grow)
            elif grow != 0:
                raise ValueError(grow)
            candidates.append((entry['score'], mask))
        painted = [(score, encode(mask)) for score, mask, _ in
                   paint_panoptic(candidates, min_area=min_area)]
        for conf in thresholds:
            predicted = [rle for score, rle in painted if score >= conf]
            for _, truth in passes:
                totals[conf].append(pq_rle(predicted, truth))
    return [dict(aggregate_pq(totals[c]), conf=c, grow=grow, min_area=min_area)
            for c in thresholds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--state", default=None,
                        help="optional state-dict checkpoint applied to the explicit base weights")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    records, split = records_for(args.annotations)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    model = YOLO(args.weights)
    state_hash = None
    state_epoch = None
    if args.state:
        state_hash = hashlib.sha256(Path(args.state).read_bytes()).hexdigest()
        saved = torch.load(args.state, map_location="cpu", weights_only=True)
        state_epoch = saved.get("epoch")
        model.model.load_state_dict(saved["state"], strict=True)
    model.model.eval(); model.model.requires_grad_(False)
    report = dict(checkpoint_sha256=original_hash, state_sha256=state_hash,
                  state_epoch=state_epoch, split=split.summary(),
                  annotation_sha256=hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                  device="cpu", variants={})
    if args.variants[0] != "identity":
        raise ValueError("identity baseline must run first")
    for variant in args.variants:
        cache = {}
        started = time.perf_counter()
        for i, name in enumerate(sorted(records)):
            image = cv2.imread(str(Path(args.images) / name))
            if image is None or image.shape[:2] != (2048, 2048):
                raise ValueError(f"invalid native photograph: {name}")
            with torch.inference_mode():
                result = model.predict(transform(image, variant), imgsz=2048,
                                       conf=.05, iou=.60, max_det=100, retina_masks=True,
                                       device="cpu", verbose=False)[0]
            entries = []
            if result.masks is not None:
                for score, mask in zip(result.boxes.conf.cpu().numpy(), result.masks.data.cpu().numpy()):
                    mask = inverse_mask(mask.astype(np.uint8), variant)
                    if mask.sum() >= 40:
                        entries.append(dict(score=float(score), rle=encode(mask)))
            cache[name] = entries
            if (i + 1) % 10 == 0:
                print(variant, i + 1, "/", len(records), flush=True)
        (output / f"{variant}_candidates.json").write_text(json.dumps(cache))
        fixed, per_record = evaluate(cache, records)
        if variant == "identity" and abs(fixed["pq"] - BASELINE_PQ) > .001:
            (output / "baseline_mismatch.json").write_text(json.dumps(fixed, indent=2))
            raise RuntimeError(f"Baseline mismatch: {fixed['pq']}; historical {BASELINE_PQ}")
        sweep = [evaluate(cache, records, conf, grow)[0]
                 for conf in (.25, .30, .35, .40, .45) for grow in (-1, 0)]
        report["variants"][variant] = dict(fixed=fixed, best=max(sweep,key=lambda r:r["pq"]),
                                          sweep=sweep, seconds=time.perf_counter()-started)
        (output / f"{variant}_records.json").write_text(json.dumps(per_record))
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        print(variant, json.dumps(report["variants"][variant]["best"]), flush=True)
    if hashlib.sha256(Path(args.weights).read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("checkpoint bytes changed")


if __name__ == "__main__":
    main()
