"""Fine-tune exp_002 at 2048 on rasterisation-corrected targets.

What this run tests
-------------------
Every detector trained on this project has been fitted to targets 10.8% larger in
area than what the scorer measures: Ultralytics rasterises with cv2.fillPoly, the
competition scores with pycocotools, and on a filament — nearly all perimeter —
that convention gap is large. Measured target-versus-scorer agreement is IoU
0.898.

A half-pixel inward polygon offset closes it to 0.959, with the area ratio moving
from 1.111 to 0.986. That correction lives in coordinate space, so unlike the 1px
erosion it has no quantisation floor, and unlike a post-hoc trim it is applied
per instance by the geometry rather than as one constant everywhere.

This run retrains on those corrected targets. It supersedes the mask_ratio=1
experiment this file previously carried: finer loss supervision against a target
that is 11% too fat reproduces the fat target more faithfully, so the defect was
never in the supervision resolution.

No wall-clock budget. Ultralytics writes a checkpoint every epoch, and every run
on this project that was stopped by a clock was still improving when it stopped.
Training ends on `patience` or on the pod being torn down, and the checkpoint
survives either way.
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
    parser.add_argument("--patience", type=int, default=40,
                        help="epochs without improvement before stopping. There is no\n"
                             "time budget: every clock-stopped run on this project was\n"
                             "still improving when its clock ran out.")
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {name}, {total:.1f} GB", flush=True)

    # 24 GB should hold batch 4 at 2048 for the medium backbone, and a 48 GB card
    # twice that. "Should" is doing work in that sentence: on a 14.56 GB T4 this
    # configuration ran out of memory at batch 8, and the run died rather than
    # retrying smaller because the fallback checked for the wrong exception. On
    # paid time an unhandled OOM is money spent for nothing, so the ladder is
    # explicit and each rung is attempted in turn.
    first = args.batch * 2 if total >= 40 else args.batch
    ladder = [b for b in (first, first // 2, first // 4) if b >= 1]
    print(f"batch ladder {ladder} at imgsz {args.imgsz}, stopping on patience {args.patience}", flush=True)

    started = time.time()
    for position, batch in enumerate(ladder):
        try:
            print(f"\n=== attempt {position + 1}/{len(ladder)}: batch {batch}, "
                  f"{remaining:.2f} h left ===", flush=True)
            train_once(YOLO(args.weights), args.data, args.imgsz, batch,
                       args.patience, args.out)
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if "out of memory" not in str(error).lower() or position == len(ladder) - 1:
                raise
            print(f"OOM at batch {batch}; retrying smaller", flush=True)
            torch.cuda.empty_cache()

    print(f"trained for {(time.time() - started) / 3600:.2f} h", flush=True)

    weights = Path(args.out) / "ft2048" / "weights"
    for candidate in ("best.pt", "last.pt"):
        path = weights / candidate
        if path.exists():
            print(f"{candidate}: {path.stat().st_size / 1e6:.1f} MB", flush=True)


def train_once(model, data, imgsz, batch, patience, out):
    model.train(
        data=data,
        imgsz=imgsz,
        batch=batch,
        epochs=1000,              # a ceiling; patience decides
        project=out,
        name="ft2048",
        exist_ok=True,
        # Fine-tuning converged weights: a gentle schedule refines them, a cold
        # start's learning rate would knock them out of their basin.
        lr0=0.0005,
        warmup_epochs=1.0,
        val=True,
        # The point of the run. Default 4 supervises masks at a quarter of the
        # input resolution; at 2048 that is a 512px target for barbs a few
        # pixels wide, and the boundary is never seen at the scale it exists at.
        # Filaments are disjoint by construction, so encoding every instance
        # into one layer with index-resolved overlaps is the wrong
        # representation and lets neighbouring boundaries bleed.
        amp=True,
        workers=8,
        cache=False,              # 974 images at 2048 will not fit in pod RAM
        # Grayscale full-disk images with no canonical up direction.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        fliplr=0.5, flipud=0.5, degrees=15.0,
        mosaic=0.0,               # pasting four disks into a frame invents limbs
        patience=patience,
        seed=2026,
        verbose=True,
    )

if __name__ == "__main__":
    main()
