"""Score the refiner against the baseline on real detector output.

The refiner was trained on synthetic degradation. That makes its own validation
IoU — 0.8529 from a coarse input of 0.7208 — a measurement of how well it undoes
*simulated* damage, not evidence that it helps on masks the detector actually
produced. This is the honest test: run the detector, refine its output, and score
both against ground truth with the competition's metric.

The mask threshold is swept because the refiner outputs probabilities, and where
that field is cut trades the same boundary the erosion constant used to trade —
but per pixel, which is the entire point of the stage.

The baseline is the shipped configuration: 1px erosion at confidence 0.35, which
scores validation PQ 0.4404 and leaderboard 0.36. The refiner replaces the
erosion rather than stacking with it; both correct the same fat-mask bias and
applying both would double-count.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

from experiments.exp_002_yolo_seg.src.predict import masks_for_image
from experiments.exp_008_refiner.src.apply import refine_instance
from experiments.exp_008_refiner.src.train_refiner import Refiner
from shared.data_split import assert_disjoint, make_split
from shared.utils import aggregate_pq, compute_pq, paint_panoptic

FULL = 2048


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
    parser.add_argument("--detector", required=True)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--out", required=True)
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
    photographs = sorted({s for _, s, _ in records})
    print(f"{len(records)} records over {len(photographs)} photographs", flush=True)

    from ultralytics import YOLO

    detector = YOLO(args.detector)
    checkpoint = torch.load(args.refiner, map_location="cpu")
    model = Refiner(checkpoint.get("width", 32))
    model.load_state_dict(checkpoint["state"])
    device = torch.device("cpu")
    model = model.to(device).eval()

    images_dir = Path(args.images)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresholds = (0.4, 0.5, 0.6)

    raw_by_stem, refined_by_stem = {}, {t: {} for t in thresholds}
    agreement = []

    for position, stem in enumerate(photographs, start=1):
        image = cv2.imread(str(images_dir / stem), cv2.IMREAD_GRAYSCALE)
        # grow=0: the refiner replaces the erosion, it does not stack with it.
        candidates = masks_for_image(detector, images_dir / stem, args.imgsz,
                                     args.conf, 0.60, 100, args.min_area, False, 0)
        raw_by_stem[stem] = candidates
        for threshold in thresholds:
            out = []
            for score, coarse in candidates:
                new = refine_instance(model, device, image, coarse, threshold)
                if int(new.sum()) >= args.min_area:
                    out.append((score, new))
                    if threshold == 0.5:
                        union = int((new | coarse).sum())
                        agreement.append(int((new & coarse).sum()) / union if union else 1.0)
            refined_by_stem[threshold][stem] = out
        if position % 10 == 0 or position == len(photographs):
            print(f"  {position}/{len(photographs)}", flush=True)

    truths = {i: gt_masks(a) for i, _s, a in records}

    def score(by_stem, erode: bool):
        rows = []
        for image_id, stem, _ in records:
            cands = []
            for s, m in by_stem[stem]:
                mm = cv2.erode(m, kernel) if erode else m
                if int(mm.sum()) >= args.min_area:
                    cands.append((s, mm))
            painted = paint_panoptic(cands, min_area=args.min_area) if cands else []
            rows.append(compute_pq([m for _, m, _ in painted], truths[image_id]))
        return aggregate_pq(rows)

    baseline = score(raw_by_stem, erode=True)     # the shipped configuration
    no_erode = score(raw_by_stem, erode=False)
    print(f"\n{'configuration':>34}{'PQ':>9}{'SQ':>8}{'RQ':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    for name, r in (("detector + 1px erosion (shipped)", baseline),
                    ("detector, no erosion", no_erode)):
        print(f"{name:>34}{r['pq']:>9.4f}{r['sq']:>8.4f}{r['rq']:>8.4f}"
              f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}", flush=True)

    results = {}
    for threshold in thresholds:
        r = score(refined_by_stem[threshold], erode=False)
        results[threshold] = r
        print(f"{'detector + refiner @ ' + str(threshold):>34}{r['pq']:>9.4f}{r['sq']:>8.4f}"
              f"{r['rq']:>8.4f}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}", flush=True)

    best_threshold = max(results, key=lambda t: results[t]["pq"])
    best = results[best_threshold]
    print(f"\nmean IoU(refined, coarse) = {np.mean(agreement):.4f}", flush=True)
    print("  near 1.0 would mean the refiner is a no-op regardless of PQ", flush=True)
    print(f"best refined PQ {best['pq']:.4f} at threshold {best_threshold} "
          f"vs baseline {baseline['pq']:.4f} ({best['pq'] - baseline['pq']:+.4f})", flush=True)

    Path(args.out).write_text(json.dumps({
        "baseline": baseline, "no_erode": no_erode,
        "refined": best, "best_threshold": best_threshold,
        "by_threshold": {str(k): v for k, v in results.items()},
        "mean_iou_refined_vs_coarse": float(np.mean(agreement)) if agreement else None,
    }, indent=2))


if __name__ == "__main__":
    main()
