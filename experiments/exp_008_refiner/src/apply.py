"""Apply the trained refiner to detector output and write a submission.

Tiling, and why it is not optional
----------------------------------
The refiner is trained on 256px crops at native resolution, and native
resolution is the entire point: the rim detail it exists to recover is a few
pixels wide at 2048 and vanishes under any downscale. Filaments, however, reach
28000px in area and are long and curved, so many do not fit in a 256 window.

Resizing a large instance into the window would discard exactly the detail the
stage was built to add. So instances are covered with overlapping tiles at
native scale and the refined probabilities are averaged where tiles overlap,
which also removes the seams a hard tile boundary would leave along a filament
running diagonally across two windows.

Each tile receives the image and the coarse mask *of the whole instance*, not
just the part inside the tile, so a filament crossing a tile edge is not
mistaken for one that ends there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.exp_002_yolo_seg.src.predict import masks_for_image
from experiments.exp_008_refiner.src.model import Refiner
from shared.utils import check_no_overlap, paint_panoptic, write_submission

FULL = 2048
CROP = 256
STRIDE = 192          # 64px of overlap; enough to blend a seam without tripling cost


def tiles_for(mask: np.ndarray):
    """Tile origins covering the mask's bounding box, clipped to the frame."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    origins = []
    y = max(0, y0 - 16)
    while True:
        x = max(0, x0 - 16)
        while True:
            origins.append((min(y, FULL - CROP), min(x, FULL - CROP)))
            if x + CROP >= x1 + 16:
                break
            x += STRIDE
        if y + CROP >= y1 + 16:
            break
        y += STRIDE
    return sorted(set(origins))


def refine_instance(model, device, image: np.ndarray, coarse: np.ndarray,
                    threshold: float) -> np.ndarray:
    """Return the refined mask for one instance."""
    accumulated = np.zeros((FULL, FULL), np.float32)
    counts = np.zeros((FULL, FULL), np.float32)

    origins = tiles_for(coarse)
    if not origins:
        return coarse

    batch = []
    for y, x in origins:
        sl = (slice(y, y + CROP), slice(x, x + CROP))
        stacked = np.stack([image[sl].astype(np.float32) / 255.0,
                            coarse[sl].astype(np.float32)])
        batch.append(stacked)

    with torch.no_grad():
        tensor = torch.from_numpy(np.stack(batch)).to(device)
        probability = torch.sigmoid(model(tensor)).cpu().numpy()[:, 0]

    for (y, x), prob in zip(origins, probability):
        sl = (slice(y, y + CROP), slice(x, x + CROP))
        accumulated[sl] += prob
        counts[sl] += 1.0

    averaged = np.divide(accumulated, counts, out=np.zeros_like(accumulated), where=counts > 0)
    refined = (averaged > threshold).astype(np.uint8)

    # The refiner adjusts a boundary; it does not relocate an instance. Anything
    # it lights up far from the coarse mask is a neighbouring filament leaking
    # into the tile, so the result is confined to a dilation of the input.
    reach = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    return (refined & reach).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="detector")
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--grow", type=int, default=0,
                        help="erosion applied to the DETECTOR mask before refining; "
                             "0 by default because the refiner is trained to correct "
                             "the fat-mask bias itself and doing both would double-count")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO

    detector = YOLO(args.weights)
    checkpoint = torch.load(args.refiner, map_location="cpu")
    model = Refiner(checkpoint.get("width", 32))
    model.load_state_dict(checkpoint["state"])
    device = torch.device("cpu")
    model = model.to(device).eval()
    print(f"refiner loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)",
          flush=True)

    paths = sorted(Path(args.images).glob("*.jpeg"))
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    predictions = {}
    changed = []
    for position, path in enumerate(paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        candidates = masks_for_image(detector, path, args.imgsz, args.conf, 0.60,
                                     100, args.min_area, False, args.grow)
        refined = []
        for score, coarse in candidates:
            new = refine_instance(model, device, image, coarse, args.threshold)
            if int(new.sum()) < args.min_area:
                continue
            inter = int((new & coarse).sum())
            union = int((new | coarse).sum())
            changed.append(inter / union if union else 1.0)
            refined.append((score, new))

        painted = paint_panoptic(refined, min_area=args.min_area) if refined else []
        predictions[path.stem] = [m for _, m, _ in painted]
        if position % 20 == 0 or position == len(paths):
            total = sum(len(v) for v in predictions.values())
            print(f"  {position}/{len(paths)} images, {total} instances", flush=True)

    if changed:
        print(f"\nmean IoU(refined, coarse) = {np.mean(changed):.4f}", flush=True)
        print("  1.0 would mean the refiner is a no-op; well below 1.0 means it is "
              "genuinely moving boundaries.", flush=True)

    empty = sum(1 for v in predictions.values() if not v)
    if empty:
        print(f"WARNING: {empty} images have no predicted instance", flush=True)

    frame = write_submission(predictions, args.output)
    check_no_overlap(args.output)
    print(f"wrote {args.output}: {len(frame)} rows over {len(predictions)} images", flush=True)


if __name__ == "__main__":
    main()
