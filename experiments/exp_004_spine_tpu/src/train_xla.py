"""Train a high-resolution segmentation model across all 8 TPU cores via SPMD.

Why SPMD rather than xmp.spawn
------------------------------
Kaggle's TPU VM is a v5litepod-8 that exports both a legacy XRT_TPU_CONFIG and
PJRT_DEVICE=TPU. Under that configuration `xmp.spawn` dies during TPU init with
"Expected 8 worker addresses, got 1", and with a BrokenProcessPool once the XRT
variable is removed — multiprocessing is simply not viable here.

It is also unnecessary. A single process on this VM already sees every core:
`xr.global_runtime_device_count()` returns 8 and the devices enumerate as xla:0
through xla:7. SPMD shards each batch across them from one process, which needs
no launcher, no distributed sampler, and no rank-aware checkpointing, and leaves
one ordinary Python process to reason about.

Sharp edges of XLA this is written around
-----------------------------------------
- Each distinct tensor shape triggers a recompilation, so batches are fixed size
  and the trailing short batch of an epoch is dropped.
- `.item()` forces the graph to execute and stalls the pipeline; running losses
  accumulate on device and are read once per epoch.
- The global batch must divide by the device count, or sharding fails.
- Checkpoints go to /kaggle/working. exp_003 lost 1.5 h because its checkpoints
  were on /kaggle/temp, which is not part of the kernel output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


class CachedSet(Dataset):
    """Reads the memmapped 1024px cache built by prepare.py."""

    def __init__(self, root: str, fold: str, augment: bool):
        self.root, self.fold, self.augment = Path(root), fold, augment
        self._image = self._mask = None
        self.length = len(json.loads((self.root / f"{fold}_ids.json").read_text()))

    def _open(self):
        # Opened lazily per worker: a memmap inherited through fork shares its
        # file offset across workers and yields corrupted reads under load.
        if self._image is None:
            self._image = np.load(self.root / f"{self.fold}_image.npy", mmap_mode="r")
            self._mask = np.load(self.root / f"{self.fold}_mask.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        self._open()
        image = np.asarray(self._image[index], dtype=np.float32) / 255.0
        mask = np.asarray(self._mask[index], dtype=np.float32)

        if self.augment:
            # Flips and right-angle rotations are exact: no interpolation, and
            # the annotation convention has no canonical up direction.
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                image, mask = image[::-1], mask[::-1]
            k = np.random.randint(4)
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            image = np.clip(image * np.random.uniform(0.85, 1.15), 0.0, 1.0)

        image = np.repeat(np.ascontiguousarray(image)[None], 3, 0)
        return torch.from_numpy(image), torch.from_numpy(np.ascontiguousarray(mask))[None]


def build_model(encoder: str):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=1)


def dice_bce(logits, target, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    union = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16, help="global batch; must divide by device count")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-budget", type=float, default=7.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    xr.use_spmd()
    n_devices = xr.global_runtime_device_count()
    device = xm.xla_device()
    print(f"SPMD across {n_devices} devices, global batch {args.batch}", flush=True)
    if args.batch % n_devices:
        raise SystemExit(f"batch {args.batch} must divide by {n_devices} devices")

    # Data-parallel mesh: shard the batch dimension, replicate everything else.
    mesh = Mesh(np.arange(n_devices), (n_devices, 1, 1, 1), ("data", "c", "h", "w"))

    train_loader = DataLoader(CachedSet(args.cache, "train", True), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(CachedSet(args.cache, "val", False), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers, drop_last=True)

    model = build_model(args.encoder).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    pos_weight = torch.tensor(args.pos_weight, device=device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        running, batches = torch.zeros((), device=device), 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            xs.mark_sharding(images, mesh, ("data", "c", "h", "w"))
            xs.mark_sharding(masks, mesh, ("data", "c", "h", "w"))

            optimiser.zero_grad()
            loss = dice_bce(model(images), masks, pos_weight)
            loss.backward()
            optimiser.step()
            xm.mark_step()
            running += loss.detach()
            batches += 1

        scheduler.step()

        model.eval()
        val_running, val_batches = torch.zeros((), device=device), 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                xs.mark_sharding(images, mesh, ("data", "c", "h", "w"))
                xs.mark_sharding(masks, mesh, ("data", "c", "h", "w"))
                val_running += dice_bce(model(images), masks, pos_weight).detach()
                xm.mark_step()
                val_batches += 1

        train_loss = (running / max(batches, 1)).item()
        val_loss = (val_running / max(val_batches, 1)).item()
        elapsed_h = (time.time() - started) / 3600
        print(f"epoch {epoch + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| {elapsed_h * 60:.1f} min", flush=True)

        # Saved every epoch so an interrupted session keeps its work. CPU tensors,
        # so the checkpoint loads without a TPU present.
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state, out_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, out_dir / "best.pt")
            print(f"  new best {best:.4f}", flush=True)

        if elapsed_h > args.time_budget:
            print(f"time budget reached at epoch {epoch + 1}", flush=True)
            break

    print(f"finished. best val loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
