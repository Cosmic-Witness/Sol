"""Run the exp_004 model over the test images and write a submission.

Inference is single-core: 180 images is small, and spawning eight replicas to
share it costs more in compilation than it saves in wall clock.

Masks are predicted at the training resolution and the *probability field* is
upsampled to 2048 before thresholding. Thresholding first and resizing the
binary mask quantises the boundary twice, and Panoptic Quality's segmentation
term is a boundary measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.exp_004_spine_tpu.src.decompose import connected_components
from experiments.exp_004_spine_tpu.src.train_xla import build_model
from shared.utils import check_no_overlap, write_submission

FULL = 2048


def predict_one(model, device, path: Path, size: int, threshold: float, tta: bool) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise SystemExit(f"cannot read {path}")
    small = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.repeat((small.astype(np.float32) / 255.0)[None], 3, 0))[None]
    tensor = tensor.to(device)

    with torch.no_grad():
        accumulated = torch.sigmoid(model(tensor))
        if tta:
            # Flips are exact operations, so averaging over them adds no
            # interpolation error while cancelling some boundary noise.
            for dims in ((2,), (3,), (2, 3)):
                flipped = torch.flip(tensor, dims=dims)
                accumulated = accumulated + torch.flip(torch.sigmoid(model(flipped)), dims=dims)
            accumulated = accumulated / 4.0

    probability = accumulated[0, 0].cpu().numpy().astype(np.float32)
    upscaled = cv2.resize(probability, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
    return (upscaled > threshold).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=150)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    except Exception:  # noqa: BLE001
        device = torch.device("cpu")
    print(f"inference device: {device}", flush=True)

    model = build_model(args.encoder)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    paths = sorted(Path(args.images).glob("*.jpeg"))
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    predictions = {}
    for position, path in enumerate(paths, start=1):
        mask = predict_one(model, device, path, args.size, args.threshold, args.tta)
        predictions[path.stem] = connected_components(mask, min_area=args.min_area)
        if position % 20 == 0 or position == len(paths):
            total = sum(len(v) for v in predictions.values())
            print(f"  {position}/{len(paths)} images, {total} instances", flush=True)

    empty = sum(1 for v in predictions.values() if not v)
    if empty:
        print(f"WARNING: {empty} images have no predicted instance", flush=True)

    frame = write_submission(predictions, args.output)
    check_no_overlap(args.output)
    print(f"wrote {args.output}: {len(frame)} rows over {len(predictions)} images", flush=True)


if __name__ == "__main__":
    main()
