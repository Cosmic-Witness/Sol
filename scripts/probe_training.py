"""Exercise crowded real training records, optimizer state and EMA validation.

Run each invocation in its own process. This is a resource probe, not a PQ run.
Uses the prepared canonical dataset and an explicit existing checkpoint.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 2 or args.batch < 1:
        parser.error("at least two optimizer steps and a positive batch are required")
    output = Path(args.output)
    if output.exists():
        parser.error("output directory already exists; keep previous probe evidence")
    if not Path(args.weights).is_file():
        parser.error("an existing checkpoint is required")

    import torch
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.torch_utils import autocast
    from experiments.exp_032_chunked.src.train import install_criterion

    expected = dict(imgsz=args.imgsz, batch=args.batch, epochs=1, mask_ratio=1,
                    overlap_mask=True, optimizer="AdamW", cos_lr=True,
                    mosaic=0., mixup=0., copy_paste=0., freeze=0)
    trainer = SegmentationTrainer(overrides=dict(
        **expected, model=args.weights, data=args.data, device=args.device,
        project=str(output.parent), name=output.name, workers=0, plots=False,
        seed=2026, lr0=5e-4, hsv_h=0., hsv_s=0., hsv_v=.15, fliplr=.5,
        flipud=.5, degrees=15., cache=False,
    ))
    trainer.add_callback("on_pretrain_routine_end", lambda t: install_criterion(t, args.chunk_size, expected))
    trainer._setup_train()
    dataset = trainer.train_loader.dataset
    crowded = sorted(range(len(dataset)), key=lambda i: len(dataset.labels[i]["cls"]), reverse=True)
    report = dict(status="running", device=str(trainer.device),
                  gpu=torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None,
                  settings=expected, steps=[])
    report_path = output / "probe.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def sync():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()

    save()
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.model.train()
    trainer.optimizer.zero_grad()
    for step in range(args.steps):
        indices = [crowded[(step * args.batch + j) % len(crowded)] for j in range(args.batch)]
        start = time.perf_counter()
        batch = trainer.preprocess_batch(dataset.collate_fn([dataset[i] for i in indices]))
        with autocast(trainer.amp):
            loss, _ = trainer.model(batch)
            loss = loss.sum()
        trainer.scaler.scale(loss).backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None):
            raise RuntimeError("non-finite loss or gradient")
        trainer.optimizer_step()
        sync()
        report["steps"].append(dict(
            seconds=time.perf_counter()-start, loss=float(loss.detach()),
            images=batch["im_file"], instances=len(batch["cls"]),
            original_instances=[len(dataset.labels[i]["cls"]) for i in indices],
            mask_shape=list(batch["masks"].shape),
        ))
        save()
    if not trainer.optimizer.state:
        raise RuntimeError("optimizer state was not allocated")
    # Evaluate the same worst training batch through EMA, including FP16 on GPU,
    # exactly as the training validator does. This is a memory check, not scoring.
    ema = trainer.ema.ema
    ema = ema.half() if trainer.amp else ema.float()
    ema.eval()
    batch["img"] = batch["img"].half() if trainer.amp else batch["img"].float()
    with torch.inference_mode():
        val_loss, _ = ema.loss(batch)
    sync()
    if not torch.isfinite(val_loss).all():
        raise RuntimeError("non-finite EMA validation loss")
    report.update(status="complete", ema_loss=val_loss.tolist(),
                  optimizer_states=len(trainer.optimizer.state),
                  cuda_peak_bytes=torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None)
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
