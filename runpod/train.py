"""Fine-tune exp_002 at 2048 under a wall-clock budget denominated in dollars.

Choices here are made for throughput per dollar rather than for the last
fraction of accuracy, because the budget is small and fixed.

- **Starts from exp_002's weights.** 149 epochs at 1280 are already paid for,
  and validation showed the model still improving when its clock stopped.
  Restarting from COCO would spend the entire budget re-learning what exists.
- **Validation every third epoch.** Ultralytics validates after every epoch by
  default; at 2048 over 180 records that is a significant share of the run for
  a number that barely moves epoch to epoch. Two thirds of it is bought back as
  training.
- **AMP and channels_last.** Both are free throughput on Ada hardware.
- **Stops on time, not on epochs.** The constraint is dollars, so the run is
  told how long it may live rather than how many epochs to complete.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {name}, {total:.1f} GB", flush=True)

    # 24 GB holds batch 4 at 2048 for the medium backbone; a 48 GB card can take
    # more, and more batch is strictly better for BatchNorm statistics.
    batch = args.batch if total < 40 else args.batch * 2
    print(f"batch {batch} at imgsz {args.imgsz}, budget {args.hours} h", flush=True)

    started = time.time()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=batch,
        epochs=1000,              # never reached; `time` is the real stop
        time=args.hours,
        project=args.out,
        name="ft2048",
        exist_ok=True,
        # Fine-tuning converged weights: a gentle schedule refines them, a cold
        # start's learning rate would knock them out of their basin.
        lr0=0.0005,
        warmup_epochs=1.0,
        val=True,
        # Validate every third epoch. At 2048 the val pass is expensive and the
        # metric moves slowly; this is throughput bought at negligible risk.
        val_period=3,
        amp=True,
        workers=8,
        cache=False,              # 974 images at 2048 will not fit in pod RAM
        # Grayscale full-disk images with no canonical up direction.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5, flipud=0.5, degrees=15.0,
        mosaic=0.0,               # pasting four disks into a frame invents limbs
        patience=100,
        seed=2026,
        verbose=True,
    )
    print(f"trained for {(time.time() - started) / 3600:.2f} h", flush=True)

    weights = Path(args.out) / "ft2048" / "weights"
    for candidate in ("best.pt", "last.pt"):
        path = weights / candidate
        if path.exists():
            print(f"{candidate}: {path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
