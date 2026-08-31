"""Train a high-resolution segmentation model on Kaggle's TPU via torch_xla.

Why a dense model, and why TPU
------------------------------
Ultralytics carries no torch_xla support — its `select_device` accepts only cpu,
cuda and mps — so YOLO cannot use a TPU at all. Continuing on TPU means a dense
architecture, and the ablation in `spine_ablation.py` shows that is not the
compromise it first appears: connected components recover ground-truth instances
at PQ 1.000 when the mask is accurate. Decomposition was never the bottleneck.
Mask precision is, and mask precision is bought with resolution, capacity and
epochs — all of which a TPU supplies.

Sharp edges of XLA that this file is written around
---------------------------------------------------
- Every distinct tensor shape triggers a fresh compilation, so batches are of
  fixed size and the final short batch of an epoch is dropped rather than
  padded.
- `.item()`, `print(loss)` and any host read force a graph execution and stall
  the pipeline. Running losses are accumulated on device and read once per
  epoch.
- Gradient updates go through `xm.optimizer_step`, which performs the
  cross-replica all-reduce; a bare `optimizer.step()` silently trains eight
  independent models.
- Checkpoints are written by the master ordinal only, with `xm.save`, and land
  under /kaggle/working so an interrupted session keeps its epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


class CachedSet(Dataset):
    """Reads the memmapped cache built by prepare.py.

    The arrays are opened lazily per worker: a memmap captured in the parent and
    inherited through fork gives every worker the same file offset and produces
    silently corrupted reads under load.
    """

    def __init__(self, root: str, fold: str, augment: bool):
        self.root, self.fold, self.augment = Path(root), fold, augment
        self._image = self._mask = None
        self.length = len(json.loads((self.root / f"{fold}_ids.json").read_text()))

    def _open(self):
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
            # Flips and 90-degree rotations only: they are exact, they need no
            # interpolation, and the annotation convention has no up direction.
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                image, mask = image[::-1], mask[::-1]
            k = np.random.randint(4)
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            image = np.clip(image * np.random.uniform(0.85, 1.15), 0.0, 1.0)

        image = np.ascontiguousarray(image)[None]          # 1 x H x W
        return torch.from_numpy(np.repeat(image, 3, 0)), torch.from_numpy(
            np.ascontiguousarray(mask))[None]


def build_model(encoder: str):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=1)


def dice_bce(logits, target, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    union = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice


def _worker(index: int, flags: dict):
    torch.manual_seed(flags["seed"])
    device = xm.xla_device()

    train_set = CachedSet(flags["cache"], "train", augment=True)
    val_set = CachedSet(flags["cache"], "val", augment=False)

    def loader_for(dataset, shuffle):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(),
            shuffle=shuffle, drop_last=True)
        return DataLoader(dataset, batch_size=flags["batch"], sampler=sampler,
                          num_workers=flags["workers"], drop_last=True)

    train_loader, val_loader = loader_for(train_set, True), loader_for(val_set, False)

    model = build_model(flags["encoder"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=flags["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=flags["epochs"])
    pos_weight = torch.tensor(flags["pos_weight"], device=device)

    checkpoint_dir = Path(flags["out"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    started = time.time()

    for epoch in range(flags["epochs"]):
        model.train()
        running, batches = torch.zeros((), device=device), 0
        for images, masks in pl.MpDeviceLoader(train_loader, device):
            optimiser.zero_grad()
            loss = dice_bce(model(images), masks, pos_weight)
            loss.backward()
            xm.optimizer_step(optimiser)
            running += loss.detach()      # stays on device; no host sync
            batches += 1
        scheduler.step()

        model.eval()
        val_running, val_batches = torch.zeros((), device=device), 0
        with torch.no_grad():
            for images, masks in pl.MpDeviceLoader(val_loader, device):
                val_running += dice_bce(model(images), masks, pos_weight).detach()
                val_batches += 1

        # One host read per epoch, after the graph has run.
        train_loss = xm.mesh_reduce("tl", (running / max(batches, 1)).item(), np.mean)
        val_loss = xm.mesh_reduce("vl", (val_running / max(val_batches, 1)).item(), np.mean)

        if xm.is_master_ordinal():
            elapsed = (time.time() - started) / 60
            print(f"epoch {epoch + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| {elapsed:.1f} min", flush=True)

        if val_loss < best:
            best = val_loss
            xm.save(model.state_dict(), str(checkpoint_dir / "best.pt"))
        xm.save(model.state_dict(), str(checkpoint_dir / "last.pt"))

        if (time.time() - started) / 3600 > flags["time_budget"]:
            if xm.is_master_ordinal():
                print(f"time budget reached at epoch {epoch + 1}", flush=True)
            break

    if xm.is_master_ordinal():
        print(f"finished. best val loss {best:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-budget", type=float, default=7.0)
    args = parser.parse_args()

    os.environ.setdefault("PJRT_DEVICE", "TPU")
    xmp.spawn(_worker, args=(vars(args),))


if __name__ == "__main__":
    main()
