"""Explicit opt-in to the chunked criterion; historical experiments stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

# Before torch/CUDA imports. There is no automatic configuration fallback.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def install_criterion(trainer, chunk_size, expected):
    import torch
    import ultralytics
    from shared.chunked_mask_loss import ChunkedSegmentationLoss

    if ultralytics.__version__ != "8.4.0":
        raise RuntimeError("exp_032 requires ultralytics==8.4.0; rerun parity tests before upgrading")
    resolved = vars(trainer.args)
    for key, value in expected.items():
        if resolved[key] != value:
            raise RuntimeError(f"configuration changed: {key}={resolved[key]!r}, expected {value!r}")
    trainer.model.criterion = ChunkedSegmentationLoss(trainer.model, chunk_size)
    # Validation computes loss on EMA, not on the live training model. Leaving
    # this at upstream would move the same OOM to the first validation epoch.
    if trainer.ema is not None:
        trainer.ema.ema.criterion = ChunkedSegmentationLoss(trainer.ema.ema, chunk_size)
    metadata = {
        "criterion": type(trainer.model.criterion).__name__, "chunk_size": chunk_size,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "device": str(next(trainer.model.parameters()).device), "resolved": resolved,
        "source_sha256": hashlib.sha256(Path(__file__).parents[3].joinpath("shared/chunked_mask_loss.py").read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    Path(trainer.save_dir, "resolved_experiment.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="chunked")
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=5e-4)
    args = parser.parse_args()
    if min(args.imgsz, args.batch, args.epochs, args.chunk_size) < 1:
        parser.error("image size, batch, epochs and chunk size must be positive")
    if args.device != "cpu" and not args.device.isdecimal():
        parser.error("use cpu or one CUDA device index; distributed training is not supported")
    if not Path(args.weights).is_file():
        parser.error("weights must be an explicit local checkpoint, with no download fallback")
    if Path(args.project, args.name).exists():
        parser.error("run directory exists; use a new name to preserve previous evidence")

    from ultralytics import YOLO
    expected = dict(imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
                    mask_ratio=1, overlap_mask=True, optimizer="AdamW", cos_lr=True,
                    mosaic=0.0, mixup=0.0, copy_paste=0.0, freeze=0)
    model = YOLO(args.weights)
    model.add_callback("on_pretrain_routine_end", lambda trainer: install_criterion(trainer, args.chunk_size, expected))
    model.train(
        **expected, data=args.data, project=args.project, name=args.name,
        device=args.device, workers=args.workers, lr0=args.lr0, lrf=.01,
        warmup_epochs=2, patience=args.epochs, save_period=1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=.15, fliplr=.5, flipud=.5, degrees=15,
        cache=False, seed=2026, plots=False,
    )


if __name__ == "__main__":
    main()
