"""Fine-tune exp_002 at 2048 with full-resolution mask supervision.

What this run tests, and why it is worth money
----------------------------------------------
Post-hoc erosion of a single pixel raised the leaderboard from 0.33 to 0.36 —
the largest gain of the project. That is evidence, not a trick: the model traces
filaments systematically too fat, and trimming them both tightens IoU on real
matches and drops borderline blobs below the minimum area.

Ultralytics supervises masks at `mask_ratio=4` by default — quarter resolution,
upsampled afterwards. On structures a few pixels wide that is precisely the
mechanism that would produce bloated boundaries, because the loss never sees the
edge at the scale the edge exists at. `overlap_mask=True` compounds it by
encoding every instance into one layer with overlaps resolved by index, when
filaments are disjoint and each deserves its own plane.

So this run fixes at the source what erosion corrects after the fact:
`mask_ratio=1`, `overlap_mask=False`, at 2048, continuing exp_002's weights.

It carries its own falsification test. If the mechanism is right, the optimal
post-hoc erosion should move from -1 towards 0 — a better-calibrated model needs
less trimming. If the optimum stays at -1, the fatness came from somewhere else
and this hypothesis is wrong regardless of what the score does.


Choices here are made for throughput per dollar rather than for the last
fraction of accuracy, because the budget is small and fixed.

- **Starts from exp_002's weights.** 149 epochs at 1280 are already paid for,
  and validation showed the model still improving when its clock stopped.
  Restarting from COCO would spend the entire budget re-learning what exists.
- **Validation every epoch, kept.** An earlier draft set `val_period=3` to buy
  back validation time. That key does not exist in this Ultralytics version —
  it is absent from DEFAULT_CFG_DICT — so it would have failed at launch on
  paid time. Validation costs roughly a tenth of an epoch here (180 records
  forward-only against 974 forward and backward), which is worth paying for a
  checkpoint-selection signal.
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

    # 24 GB should hold batch 4 at 2048 for the medium backbone, and a 48 GB card
    # twice that. "Should" is doing work in that sentence: on a 14.56 GB T4 this
    # configuration ran out of memory at batch 8, and the run died rather than
    # retrying smaller because the fallback checked for the wrong exception. On
    # paid time an unhandled OOM is money spent for nothing, so the ladder is
    # explicit and each rung is attempted in turn.
    first = args.batch * 2 if total >= 40 else args.batch
    ladder = [b for b in (first, first // 2, first // 4) if b >= 1]
    print(f"batch ladder {ladder} at imgsz {args.imgsz}, budget {args.hours} h", flush=True)

    started = time.time()
    for position, batch in enumerate(ladder):
        remaining = args.hours - (time.time() - started) / 3600
        if remaining <= 0.1:
            raise SystemExit("budget exhausted before training could start")
        try:
            print(f"\n=== attempt {position + 1}/{len(ladder)}: batch {batch}, "
                  f"{remaining:.2f} h left ===", flush=True)
            train_once(YOLO(args.weights), args.data, args.imgsz, batch,
                       remaining, args.out)
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


def train_once(model, data, imgsz, batch, hours, out):
    model.train(
        data=data,
        imgsz=imgsz,
        batch=batch,
        epochs=1000,              # never reached; `time` is the real stop
        time=hours,
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
        mask_ratio=1,
        # Filaments are disjoint by construction, so encoding every instance
        # into one layer with index-resolved overlaps is the wrong
        # representation and lets neighbouring boundaries bleed.
        overlap_mask=False,
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

if __name__ == "__main__":
    main()
