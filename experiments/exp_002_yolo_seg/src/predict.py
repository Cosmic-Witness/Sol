"""Turn YOLO segmentation output into a valid, pixel-disjoint submission.

Three things here differ from the public 0.55 inference notebook, and each was a
deliberate change rather than a stylistic one.

Overlaps are resolved by confidence, not by class id.
    The public notebook sorts masks with `result.boxes.cls` before painting them.
    The model is trained with `nc=1`, so every element of that array is 0 and the
    sort is a no-op: whichever mask happens to come last in the raw output wins
    the contested pixels. Panoptic Quality is decided by IoU against matched
    ground truth, so handing contested pixels to an arbitrary mask instead of the
    confident one costs IoU on both instances. `boxes.conf` is the field that
    orders them correctly.

Painting reuses `shared.utils.paint_panoptic`.
    The submission is rejected outright if any two masks in an image share a
    pixel, so the disjointness guarantee belongs in one tested place rather than
    being re-implemented per experiment.

Masks are thresholded at full resolution.
    YOLO returns mask logits at the network's stride, not at 2048. Resizing a
    binarised mask compounds its staircase edges; resizing the probability field
    and thresholding afterwards keeps the boundary where the model put it, and
    boundary placement is exactly what segmentation quality measures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from shared.utils import check_no_overlap, paint_panoptic, write_submission

FULL_SIZE = 2048


def masks_for_image(
    model,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    min_area: int,
    tta: bool,
    grow: int = 0,
) -> list[tuple[float, np.ndarray]]:
    """Predict one image; return (confidence, full-resolution mask) pairs."""
    result = model.predict(
        str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        augment=tta,
        retina_masks=True,
        verbose=False,
    )[0]

    if result.masks is None or len(result.masks.data) == 0:
        return []

    raw = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    order = np.argsort(scores)[::-1]

    # paint_panoptic consumes (score, mask) pairs and orders them itself, so the
    # confidence travels with the mask rather than being implied by list order.
    candidates = []
    for index in order:
        mask = raw[index].astype(np.float32)
        if mask.shape != (FULL_SIZE, FULL_SIZE):
            # Interpolate the soft mask, then threshold, so the edge is not
            # quantised twice.
            mask = cv2.resize(mask, (FULL_SIZE, FULL_SIZE), interpolation=cv2.INTER_LINEAR)
        binary = (mask > 0.5).astype(np.uint8)
        if grow:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1,) * 2)
            binary = (cv2.dilate if grow > 0 else cv2.erode)(binary, kernel).astype(np.uint8)
        if int(binary.sum()) < min_area:
            continue
        candidates.append((float(scores[index]), binary))
    return candidates


def predict_directory(
    model,
    images_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    min_area: int,
    tta: bool,
    grow: int = 0,
) -> dict[str, list[np.ndarray]]:
    predictions: dict[str, list[np.ndarray]] = {}
    paths = sorted(images_dir.glob("*.jpeg"))
    if not paths:
        raise SystemExit(f"no .jpeg images under {images_dir}")

    for position, path in enumerate(paths, start=1):
        candidates = masks_for_image(model, path, imgsz, conf, iou, max_det, min_area, tta, grow)
        # paint_panoptic returns (score, mask, area); write_submission wants the
        # masks alone, still in the order painting settled on.
        painted = paint_panoptic(candidates, min_area=min_area) if candidates else []
        predictions[path.stem] = [mask for _, mask, _ in painted]
        if position % 20 == 0 or position == len(paths):
            total = sum(len(v) for v in predictions.values())
            print(f"  {position}/{len(paths)} images, {total} instances", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--min-area", type=int, default=150)
    parser.add_argument("--grow", type=int, default=0,
                        help="morphological growth in px; negative erodes. "
                             "Validation puts the optimum at -1: YOLO's mask "
                             "prototypes are systematically slightly too large, "
                             "so eroding both tightens IoU on matches and culls "
                             "marginal detections below min_area.")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    predictions = predict_directory(
        model,
        Path(args.images),
        args.imgsz,
        args.conf,
        args.iou,
        args.max_det,
        args.min_area,
        args.tta,
        args.grow,
    )

    empty = [stem for stem, masks in predictions.items() if not masks]
    if empty:
        # An image with no predicted filament contributes only false negatives.
        # It is worth seeing in the log, because a large count usually means the
        # confidence threshold is too high rather than that the Sun was quiet.
        print(f"WARNING: {len(empty)} images have no predicted instance", flush=True)

    frame = write_submission(predictions, args.output)
    check_no_overlap(args.output)
    print(f"wrote {args.output}: {len(frame)} rows over {len(predictions)} images", flush=True)


if __name__ == "__main__":
    main()
