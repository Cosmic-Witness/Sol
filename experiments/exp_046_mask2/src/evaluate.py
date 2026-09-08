"""CPU validation and test inference for a completed exp046 checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from experiments.exp_033_inference.src.sweep import encode, evaluate_thresholds, records_for
from shared.utils import check_no_overlap, paint_panoptic

CURRENT_BEST_PQ = 0.45707595553373687
MASK_THRESHOLDS = (.4, .5, .6)
CONFIDENCES = (.2, .4, .6, .8)


def image_tensor(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048):
        raise ValueError(f'Invalid native photograph: {path}')
    pixels = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    pixels = np.repeat(pixels[None], 3, axis=0).astype(np.float32)/255
    pixels -= np.array([.485, .456, .406], np.float32)[:, None, None]
    pixels /= np.array([.229, .224, .225], np.float32)[:, None, None]
    return torch.from_numpy(pixels)[None]


@torch.inference_mode()
def infer(model, path, size):
    output = model(pixel_values=image_tensor(path, size))
    classes = output.class_queries_logits[0].softmax(-1)[:, 0].numpy()
    probabilities = output.masks_queries_logits[0].sigmoid().numpy()
    if not np.isfinite(classes).all() or not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite CPU inference')
    return classes, probabilities


def decode(classes, probabilities, threshold, floor=.2):
    """Use identical native resizing and mask-quality scores on both splits."""
    entries = []
    for score, probability in zip(classes, probabilities):
        if score < floor:
            continue
        probability = cv2.resize(probability, (2048,2048), interpolation=cv2.INTER_LINEAR)
        binary = probability >= threshold
        area = int(binary.sum())
        if area < 300:
            continue
        quality = float(probability[binary].mean())
        confidence = float(score)*quality
        if confidence >= floor:
            entries.append(dict(score=confidence, rle=encode(binary)))
    return entries


def paint(entries, rule):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    candidates = []
    for entry in entries:
        if entry['score'] < rule['conf']:
            continue
        mask = mu.decode(entry['rle'])
        if rule['grow'] == -1:
            mask = cv2.erode(mask, kernel)
        elif rule['grow'] != 0:
            raise ValueError('Unexpected mask growth rule')
        candidates.append((entry['score'], mask))
    return paint_panoptic(candidates, min_area=rule['min_area'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--test-images', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=0, help='Partial smoke check; never writes a submission')
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.manual_seed(2026)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['split_seed'] != 2026 or saved['loss'] != 'Hungarian class+BCE+Dice':
        raise ValueError('Unexpected checkpoint training contract')
    if not args.limit and (saved['size'] != 1024 or saved['steps'] < 974):
        raise ValueError('Full evaluation requires at least one epoch at 1024px')
    model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(saved['config']))
    model.load_state_dict(saved['model'], strict=True); model.eval()
    records, split = records_for(args.annotations)
    names = sorted(records)
    if args.limit:
        names = names[:args.limit]
        records = {name:records[name] for name in names}
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    caches = {threshold:{} for threshold in MASK_THRESHOLDS}
    for i, name in enumerate(names, 1):
        classes, probabilities = infer(model, args.images/name, saved['size'])
        for threshold in MASK_THRESHOLDS:
            caches[threshold][name] = decode(classes, probabilities, threshold)
        if i%10 == 0 or i == len(names):
            print(json.dumps(dict(event='validation_inference', completed=i, total=len(names),
                                  elapsed=time.perf_counter()-started)),flush=True)
    results = []
    for threshold, cache in caches.items():
        (args.output/f'candidates_{threshold}.json').write_text(json.dumps(cache))
        for grow in (0,-1):
            rows = evaluate_thresholds(cache, records, CONFIDENCES, grow=grow, min_area=300)
            results.extend(dict(row, mask_threshold=threshold) for row in rows)
        print(json.dumps(dict(event='threshold_scored',mask_threshold=threshold,
                              best_pq=max(row['pq'] for row in results))),flush=True)
    best = max(results, key=lambda row:row['pq'])
    report = dict(complete=not args.limit, checkpoint_sha256=checkpoint_hash,
                  annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                  device='cpu', size=saved['size'], split=split.summary(),
                  validation_photographs=len(records), sweep=results, best=best,
                  current_best_pq=CURRENT_BEST_PQ,
                  eligible_for_submission=not args.limit and best['pq'] > CURRENT_BEST_PQ,
                  elapsed_seconds=time.perf_counter()-started)
    (args.output/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(event='validation_done', best=best,
                          eligible=report['eligible_for_submission'])),flush=True)
    if not report['eligible_for_submission'] or args.test_images is None:
        return
    test_files = sorted(args.test_images.glob('*.jpeg'))
    if len(test_files) != 180:
        raise ValueError('Expected all 180 test photographs')
    csv_path = args.output/'submission.csv'
    counts = {}
    with csv_path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['filament_id','segmentation_rle'])
        writer.writeheader()
        for i, path in enumerate(test_files,1):
            classes, probabilities = infer(model, path, saved['size'])
            entries = decode(classes, probabilities, best['mask_threshold'])
            masks = paint(entries,best)
            counts[path.stem] = len(masks)
            for j, (_, mask, _) in enumerate(masks,1):
                writer.writerow(dict(filament_id=f'{path.stem}_{j}',segmentation_rle=encode(mask)['counts']))
            if i%10 == 0:
                print(json.dumps(dict(event='test_inference',completed=i,total=180)),flush=True)
    check_no_overlap(str(csv_path))
    (args.output/'submission_report.json').write_text(json.dumps(dict(
        rule=best,checkpoint_sha256=checkpoint_hash,photographs=180,
        counts=counts,rows=sum(counts.values()),
        submission_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest()),indent=2))


if __name__ == '__main__':
    main()
