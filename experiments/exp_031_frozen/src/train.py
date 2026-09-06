"""One training attempt, in its own process.

Run as a subprocess so that a failed attempt releases every byte before the next
one starts. The first version of this ladder called `YOLO(...).train()` in a loop
inside one process: `torch.cuda.empty_cache()` cannot reclaim memory the previous
trainer still holds a reference to, and the evidence was two attempts reporting
byte-identical out-of-memory errors -- 3.11 GiB wanted, 312.81 MiB free -- while
the batch size between them was halved. The second attempt never really ran.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--freeze", type=int, default=11)
    parser.add_argument("--mask-ratio", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=5e-4)
    args = parser.parse_args()

    from ultralytics import YOLO

    YOLO(args.weights).train(
        data=args.data,
        imgsz=args.imgsz, batch=args.batch,
        epochs=args.epochs, patience=args.epochs,
        project=args.project, name=args.name, exist_ok=True,
        freeze=args.freeze,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, cos_lr=True, warmup_epochs=2.0,
        mask_ratio=args.mask_ratio, overlap_mask=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5, flipud=0.5, degrees=15.0, mosaic=0.0,
        cache=False, workers=2, seed=2026, verbose=True, plots=False,
    )


if __name__ == "__main__":
    main()
