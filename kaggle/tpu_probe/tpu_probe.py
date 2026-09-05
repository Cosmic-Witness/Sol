"""Establish what a Kaggle TPU session can actually run before committing hours.

Ultralytics contains no torch_xla references and its select_device accepts only
cpu, cuda and mps, so YOLO cannot train on TPU at any resolution. Continuing on
TPU therefore means a different architecture, and the candidate is a dense
segmentation model under torch_xla.

This probe answers the questions that decide whether that is viable, cheaply:
does torch_xla import, how many cores are visible, does a real convolutional
forward and backward pass compile and run, and how long does a step take at the
resolution that matters. A wrong answer here costs ten minutes instead of a
night of quota.
"""

import os
import time

print("=" * 70, flush=True)
for key in ("TPU_NAME", "XRT_TPU_CONFIG", "PJRT_DEVICE", "TPU_WORKER_ID"):
    print(f"{key}={os.environ.get(key)}", flush=True)

try:
    import torch
    print(f"torch {torch.__version__}", flush=True)
except Exception as error:  # noqa: BLE001
    raise SystemExit(f"torch unavailable: {error}")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f"torch_xla {getattr(torch_xla, '__version__', 'unknown')}", flush=True)
except Exception as error:  # noqa: BLE001
    raise SystemExit(
        f"torch_xla import failed: {error}\n"
        "Without it there is no TPU path for a PyTorch model in this image."
    )

device = xm.xla_device()
print(f"xla device: {device}", flush=True)
try:
    print(f"world size: {xm.xrt_world_size()}", flush=True)
except Exception:  # noqa: BLE001
    print("world size: unavailable (single-process probe)", flush=True)

# A real conv stack, not a matmul: XLA compiles convolutions differently and
# this is what the segmentation model will actually do.
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 1, 1),
).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for size, batch in ((512, 8), (1024, 4)):
    try:
        images = torch.randn(batch, 3, size, size, device=device)
        target = torch.zeros(batch, 1, size // 2, size // 2, device=device)
        # First step includes XLA compilation; time the steady state separately.
        for index in range(3):
            if index == 1:
                xm.mark_step()
                start = time.time()
            optimiser.zero_grad()
            loss = loss_fn(model(images), target)
            loss.backward()
            xm.optimizer_step(optimiser)
            xm.mark_step()
        elapsed = (time.time() - start) / 2
        print(f"OK  {size}px batch {batch}: {elapsed:.3f} s/step steady state", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"FAIL {size}px batch {batch}: {type(error).__name__}: {str(error)[:200]}", flush=True)

print("=" * 70, flush=True)
print("PROBE COMPLETE", flush=True)
