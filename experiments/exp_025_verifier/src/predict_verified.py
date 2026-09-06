"""Emit a deliberately high-recall submission, to test what the test labels are.

Every negative result today was measured against validation, where each record
carries one annotator's opinion. The verifier is 72% accurate at identifying
filaments *somebody* would draw and only 17% accurate at matching the particular
person who labelled a given record, and that difference is what sank it.

If the test ground truth is more inclusive than one annotator -- a consensus, a
more thorough labeller, or simply a different person's threshold for what counts
-- then a rule that emits more would score better in public than it does locally.
This configuration is 0.041 PQ *worse* on validation than the baseline. If it
comes back at or above the baseline's public score, the local split's
annotator-specificity is the thing holding the score down, and that is worth
knowing more than the submission costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.exp_025_verifier.src.harvest import crop_of
from experiments.exp_025_verifier.src.model import Verifier
from shared.utils import check_no_overlap, paint_panoptic, write_submission

FULL = 2048


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-conf", type=float, default=0.30)
    parser.add_argument("--floor-conf", type=float, default=0.05)
    parser.add_argument("--gate", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=250)
    args = parser.parse_args()

    from ultralytics import YOLO
    detector = YOLO(args.weights)

    checkpoint = torch.load(args.verifier, map_location="cpu")
    verifier = Verifier(checkpoint.get("width", 24))
    verifier.load_state_dict(checkpoint["state"])
    verifier.eval()

    paths = sorted(Path(args.images).glob("*.jpeg"))
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    predictions: dict[str, list[np.ndarray]] = {}
    promoted_total = 0
    for position, path in enumerate(paths, start=1):
        result = detector.predict(str(path), imgsz=FULL, conf=args.floor_conf,
                                  iou=0.60, max_det=100, retina_masks=True,
                                  verbose=False)[0]
        if result.masks is None or not len(result.masks.data):
            predictions[path.stem] = []
            continue
        raw = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        photograph = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        masks, confs, batch, index_of = [], [], [], []
        for index in range(raw.shape[0]):
            mask = raw[index].astype(np.uint8)
            if mask.shape != (FULL, FULL):
                mask = (cv2.resize(mask.astype(np.float32), (FULL, FULL),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
            masks.append(mask)
            confs.append(float(scores[index]))
            cropped = crop_of(photograph, mask)
            if cropped is not None:
                patch, patch_mask = cropped
                batch.append(np.stack([patch.astype(np.float32) / 255.0,
                                       patch_mask.astype(np.float32)]))
                index_of.append(index)

        probability = np.zeros(len(masks))
        if batch:
            with torch.no_grad():
                logits = verifier(torch.from_numpy(np.stack(batch))).numpy()
            probability[index_of] = 1.0 / (1.0 + np.exp(-logits))

        candidates = []
        for index, mask in enumerate(masks):
            if confs[index] >= args.base_conf:
                candidates.append((confs[index], mask))
            elif probability[index] >= args.gate:
                # Below the emitted floor: keep it just under the floor so a
                # confident detection always paints first.
                candidates.append((args.base_conf - 0.01, mask))
                promoted_total += 1
        painted = paint_panoptic(candidates, min_area=args.min_area) if candidates else []
        predictions[path.stem] = [m for _, m, _ in painted]
        if position % 40 == 0 or position == len(paths):
            print(f"  {position}/{len(paths)}", flush=True)

    print(f"promoted by the verifier: {promoted_total}", flush=True)
    frame = write_submission(predictions, args.output)
    check_no_overlap(args.output)
    print(f"wrote {args.output}: {len(frame)} rows over {len(predictions)} images",
          flush=True)


if __name__ == "__main__":
    main()
