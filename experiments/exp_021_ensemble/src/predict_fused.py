"""Emit the fused submission: exp_002's masks, vetoed by exp_010's agreement.

The winning rule from the validation sweep, at PQ 0.4411 against 0.4404: keep
exp_002's candidates above confidence 0.35 with the one-pixel erosion it prefers;
those exp_010 also found (IoU >= 0.5 against any of its candidates at its own
operating point) are painted first, and those it did not find must reach 0.55
alone.

The margin is 0.0007 and the diagnostic behind it is that the veto removes seven
false positives out of 456, so this is not expected to move the leaderboard. It
is submitted as a datum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from shared.utils import check_no_overlap, paint_panoptic, write_submission

FULL = 2048
AGREE_IOU = 0.5
MIN_AREA = 300


def morph(binary: np.ndarray, pixels: int) -> np.ndarray:
    if not pixels:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(binary, kernel).astype(np.uint8)


def candidates(model, path: Path, conf: float, grow: int) -> list[tuple[float, np.ndarray]]:
    result = model.predict(str(path), imgsz=FULL, conf=conf, iou=0.60,
                           max_det=100, retina_masks=True, verbose=False)[0]
    if result.masks is None or not len(result.masks.data):
        return []
    raw = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    out = []
    for index in range(raw.shape[0]):
        mask = raw[index].astype(np.uint8)
        if mask.shape != (FULL, FULL):
            mask = (cv2.resize(mask.astype(np.float32), (FULL, FULL),
                               interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
        out.append((float(scores[index]), morph(mask, grow)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-a", required=True)
    parser.add_argument("--weights-b", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf-a", type=float, default=0.35)
    # The validation sweep checked agreement against exp_010's whole cached
    # pool, which nearmiss dumped at a 0.05 floor -- not against its candidates
    # at its own operating point. Running it at 0.30 here made the confirmation
    # set far smaller, vetoed 312 candidates instead of a handful, and emitted
    # 979 instances where the baseline emits 1238. That is a different rule from
    # the one measured, so the floor has to match the one the measurement used.
    parser.add_argument("--conf-b", type=float, default=0.05)
    parser.add_argument("--grow-a", type=int, default=-1)
    parser.add_argument("--grow-b", type=int, default=0)
    parser.add_argument("--veto", type=float, default=0.55)
    args = parser.parse_args()

    from ultralytics import YOLO

    a = YOLO(args.weights_a)
    b = YOLO(args.weights_b)
    paths = sorted(Path(args.images).glob("*.jpeg"))
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    predictions: dict[str, list[np.ndarray]] = {}
    confirmed_total = vetoed_total = 0
    for position, path in enumerate(paths, start=1):
        ca = candidates(a, path, args.conf_a, args.grow_a)
        cb = candidates(b, path, args.conf_b, args.grow_b)

        if ca and cb:
            left = mask_util.encode(np.asfortranarray(
                np.stack([m for _, m in ca], axis=-1)))
            right = mask_util.encode(np.asfortranarray(
                np.stack([m for _, m in cb], axis=-1)))
            agree = mask_util.iou(left, right, [0] * len(cb)).max(axis=1)
        else:
            agree = np.zeros(len(ca))

        kept = []
        for index, (score, mask) in enumerate(ca):
            if agree[index] >= AGREE_IOU:
                kept.append((score + 1.0, mask))     # confirmed: paint first
                confirmed_total += 1
            elif score >= args.veto:
                kept.append((score, mask))
            else:
                vetoed_total += 1
        painted = paint_panoptic(kept, min_area=MIN_AREA) if kept else []
        predictions[path.stem] = [m for _, m, _ in painted]
        if position % 40 == 0 or position == len(paths):
            print(f"  {position}/{len(paths)}", flush=True)

    print(f"confirmed by exp_010: {confirmed_total} | "
          f"vetoed below {args.veto}: {vetoed_total}", flush=True)
    frame = write_submission(predictions, args.output)
    check_no_overlap(args.output)
    print(f"wrote {args.output}: {len(frame)} rows over {len(predictions)} images",
          flush=True)


if __name__ == "__main__":
    main()
