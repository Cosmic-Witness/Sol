"""Isolated forward/backward mask-loss comparison (NOT a full-model fit gate).

Reports saved autograd storage and process RSS on CPU, CUDA peak allocation on
GPU. Target maps are compact; anchors_per_instance models YOLO's repeated
positive assignments. Each implementation runs in a fresh subprocess. Logs,
timeouts and non-OOM errors are retained, never interpreted as a smaller fit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def worker(args):
    import psutil
    import torch
    import ultralytics
    from types import SimpleNamespace
    from ultralytics.utils.loss import v8SegmentationLoss
    from shared.chunked_mask_loss import ChunkedSegmentationLoss

    torch.set_num_threads(4)
    torch.manual_seed(2026)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    anchors = args.instances * args.anchors_per_instance
    h = w = args.size
    masks = (torch.arange(h * w, device=device).reshape(1, h, w) % (args.instances + 1)).float()
    coefficients = torch.randn(1, anchors, 32, device=device, requires_grad=True)
    proto = torch.randn(1, 32, h, w, device=device, requires_grad=True)
    inputs = (
        torch.ones(1, anchors, dtype=torch.bool, device=device), masks,
        torch.arange(anchors, device=device).remainder(args.instances).view(1, -1),
        torch.tensor([0., 0., w, h], device=device).repeat(1, anchors, 1),
        torch.zeros(args.instances, device=device), proto, coefficients,
        torch.tensor([h, w], device=device),
    )
    if args.worker == "upstream":
        criterion = SimpleNamespace(overlap=True, single_mask_loss=v8SegmentationLoss.single_mask_loss)
        calculate = lambda: v8SegmentationLoss.calculate_segmentation_loss(criterion, *inputs)
    else:
        criterion = object.__new__(ChunkedSegmentationLoss)
        criterion.overlap, criterion.chunk_size = True, args.chunk_size
        calculate = lambda: criterion.calculate_segmentation_loss(*inputs)
    saved = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    pass_seconds = []
    for _ in range(2):
        # Report a cold and a warm pass: the first checkpoint call also pays
        # PyTorch's lazy initialization costs, which are not steady throughput.
        coefficients.grad = proto.grad = None
        saved.clear()
        start = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = calculate()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        pass_seconds.append(time.perf_counter() - start)
    if not torch.isfinite(loss) or not torch.isfinite(proto.grad).all() or not torch.isfinite(coefficients.grad).all():
        raise RuntimeError("non-finite loss or gradient")
    memory = psutil.Process().memory_info()
    return dict(
        status="ok", implementation=args.worker, device=str(device), size=h,
        instances=args.instances, positive_anchors=anchors, chunk_size=args.chunk_size,
        loss=float(loss.detach()), pass_seconds=pass_seconds,
        saved_storage_bytes=sum(saved.values()), rss_bytes=memory.rss,
        peak_rss_bytes=getattr(memory, "peak_wset", None),
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        torch=torch.__version__, ultralytics=ultralytics.__version__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--instances", type=int, default=40)
    parser.add_argument("--anchors-per-instance", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", choices=["upstream", "chunked"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.size, args.instances, args.anchors_per_instance, args.chunk_size, args.timeout) < 1:
        parser.error("sizes and timeout must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = worker(args)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return
    results = []
    for implementation in ("upstream", "chunked"):
        child = output.with_name(output.stem + f"_{implementation}.json")
        # Ignore/remove only this exact old child report; never consume stale success.
        child.unlink(missing_ok=True)
        command = [sys.executable, "-m", "scripts.probe_mask_loss", "--worker", implementation,
                   "--size", str(args.size), "--instances", str(args.instances),
                   "--anchors-per-instance", str(args.anchors_per_instance),
                   "--chunk-size", str(args.chunk_size), "--device", args.device, "--output", str(child)]
        log = child.with_suffix(".log")
        with log.open("w", encoding="utf-8") as stream:
            try:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                result = json.loads(child.read_text()) if completed.returncode == 0 else {
                    "status": "error", "returncode": completed.returncode}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "timeout_seconds": args.timeout}
        results.append(dict(result, implementation=implementation, log=str(log)))
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results[-1]), flush=True)
    if any(result["status"] != "ok" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
