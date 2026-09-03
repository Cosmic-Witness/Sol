"""Train the boundary refiner across all 8 TPU cores.

Reuses the SPMD arrangement proved in exp_004: Kaggle's v5litepod-8 breaks
xmp.spawn during TPU init, but a single process already sees all eight devices,
so the batch is sharded from one process with no launcher and no distributed
sampler.

The network takes two channels — the image crop and the coarse mask — and
returns a refined mask. Giving it the coarse mask rather than only the image is
what makes it a *refiner* rather than a second detector: it never has to decide
which filament is meant, only where that filament's boundary is.

The loss is BCE plus soft Dice, weighted towards the boundary. Interior pixels
are already correct in the coarse input and contribute nothing to learn from;
the measurement puts 64% of the error within two pixels of the rim, so the rim
is where the gradient should be spent.
"""

from __future__ import annotations

import argparse
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


class CropSet(Dataset):
    def __init__(self, root: str, fold: str, augment: bool):
        self.root, self.fold, self.augment = Path(root), fold, augment
        self._image = self._coarse = self._truth = None
        self.length = len(np.load(self.root / f"{fold}_truth.npy", mmap_mode="r"))

    def _open(self):
        if self._image is None:
            self._image = np.load(self.root / f"{self.fold}_image.npy", mmap_mode="r")
            self._coarse = np.load(self.root / f"{self.fold}_coarse.npy", mmap_mode="r")
            self._truth = np.load(self.root / f"{self.fold}_truth.npy", mmap_mode="r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open()
        image = np.asarray(self._image[index], np.float32) / 255.0
        coarse = np.asarray(self._coarse[index], np.float32)
        truth = np.asarray(self._truth[index], np.float32)

        if self.augment:
            # The disk has no canonical orientation, so all eight dihedral views
            # are valid and exact — no interpolation, no invented pixels.
            if np.random.rand() < 0.5:
                image, coarse, truth = image[:, ::-1], coarse[:, ::-1], truth[:, ::-1]
            k = np.random.randint(4)
            if k:
                image, coarse, truth = (np.rot90(image, k), np.rot90(coarse, k), np.rot90(truth, k))
            image = np.clip(image * np.random.uniform(0.85, 1.15), 0.0, 1.0)

        stacked = np.stack([np.ascontiguousarray(image), np.ascontiguousarray(coarse)])
        return torch.from_numpy(stacked), torch.from_numpy(np.ascontiguousarray(truth))[None]


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class Refiner(nn.Module):
    """A small U-Net. Capacity is deliberately modest.

    The task is local — move a boundary a pixel or two given an image and an
    approximate mask — so depth buys little and costs throughput. Every crop is
    256x256 and the object is already localised, which is precisely the
    advantage this stage has over the detector operating on a 2048 frame.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = conv_block(2, w), conv_block(w, w * 2), conv_block(w * 2, w * 4)
        self.bottleneck = conv_block(w * 4, w * 8)
        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.dec3 = conv_block(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = conv_block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = conv_block(w * 2, w)
        self.head = nn.Conv2d(w, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def boundary_weighted_loss(logits, target, coarse):
    """BCE weighted towards the rim, plus soft Dice.

    Pixels where the coarse mask and the truth already agree are not where the
    score is lost. Weighting the disagreement band up concentrates the gradient
    on the 64% of error that sits within two pixels of the boundary.
    """
    disagreement = (coarse - target).abs()
    weight = 1.0 + 4.0 * disagreement
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, weight=weight, reduction="mean")

    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    union = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64, help="global; must divide by device count")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25,
                        help="epochs without validation improvement before stopping. "
                             "There is deliberately no wall-clock budget: checkpoints "
                             "are written every epoch to /kaggle/working, which is the "
                             "kernel output, so an interrupted session keeps its work "
                             "and a clock would only stop a run that was still learning.")
    args = parser.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)

    xr.use_spmd()
    devices = xr.global_runtime_device_count()
    device = xm.xla_device()
    print(f"SPMD across {devices} devices, global batch {args.batch}", flush=True)
    if args.batch % devices:
        raise SystemExit(f"batch {args.batch} must divide by {devices}")
    mesh = Mesh(np.arange(devices), (devices, 1, 1, 1), ("data", "c", "h", "w"))

    train_loader = DataLoader(CropSet(args.data, "train", True), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(CropSet(args.data, "val", False), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers, drop_last=True)

    model = Refiner(args.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    since_improvement = 0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        running, batches = torch.zeros((), device=device), 0
        for images, truth in train_loader:
            images, truth = images.to(device), truth.to(device)
            xs.mark_sharding(images, mesh, ("data", "c", "h", "w"))
            xs.mark_sharding(truth, mesh, ("data", "c", "h", "w"))
            coarse = images[:, 1:2]
            optimiser.zero_grad()
            loss = boundary_weighted_loss(model(images), truth, coarse)
            loss.backward()
            optimiser.step()
            xm.mark_step()
            running += loss.detach()
            batches += 1
        scheduler.step()

        model.eval()
        val_running, val_batches, iou_sum, iou_n = torch.zeros((), device=device), 0, 0.0, 0
        with torch.no_grad():
            for images, truth in val_loader:
                images, truth = images.to(device), truth.to(device)
                xs.mark_sharding(images, mesh, ("data", "c", "h", "w"))
                xs.mark_sharding(truth, mesh, ("data", "c", "h", "w"))
                coarse = images[:, 1:2]
                logits = model(images)
                val_running += boundary_weighted_loss(logits, truth, coarse).detach()
                predicted = (torch.sigmoid(logits) > 0.5).float()
                inter = (predicted * truth).sum((1, 2, 3))
                union = ((predicted + truth) > 0).float().sum((1, 2, 3))
                iou_sum += (inter / union.clamp(min=1)).sum().item()
                iou_n += images.shape[0]
                xm.mark_step()
                val_batches += 1

        train_loss = (running / max(batches, 1)).item()
        val_loss = (val_running / max(val_batches, 1)).item()
        val_iou = iou_sum / max(iou_n, 1)
        elapsed = (time.time() - started) / 3600
        print(f"epoch {epoch + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val IoU {val_iou:.4f} | {elapsed * 60:.1f} min", flush=True)

        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({"state": state, "width": args.width}, out_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save({"state": state, "width": args.width}, out_dir / "best.pt")
            print(f"  new best {best:.4f} (val IoU {val_iou:.4f})", flush=True)

            since_improvement = 0
        else:
            since_improvement += 1

        if since_improvement >= args.patience:
            print(f"early stop: {args.patience} epochs without improvement "
                  f"(best {best:.4f} at epoch {epoch + 1 - since_improvement})", flush=True)
            break

    print(f"finished. best val loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
