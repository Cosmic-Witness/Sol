"""Train the candidate verifier on TPU.

The SPMD arrangement is the one proved in exp_004 and reused by exp_008: Kaggle's
v5litepod-8 breaks under `xmp.spawn`, but a single process sees all eight devices
and shards the batch across them.

Selection is by validation **average precision**, not by loss. The emission rule
that consumes this model sweeps a threshold, so what matters is the ordering it
produces over candidates, and at a ten percent base rate a loss can improve while
the ordering does not. exp_012 selected a refiner on loss and picked a checkpoint
worse than its own epoch two.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.exp_025_verifier.src.model import Verifier, positive_weighted_loss

import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


class CropSet(Dataset):
    def __init__(self, root: str, fold: str, augment: bool):
        self.root, self.fold, self.augment = Path(root), fold, augment
        self._image = self._mask = None
        self.label = np.load(self.root / f"{fold}_label.npy")

    def _open(self):
        if self._image is None:
            self._image = np.load(self.root / f"{self.fold}_image.npy", mmap_mode="r")
            self._mask = np.load(self.root / f"{self.fold}_mask.npy", mmap_mode="r")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        self._open()
        image = np.asarray(self._image[index], np.float32) / 255.0
        mask = np.asarray(self._mask[index], np.float32) / 255.0

        if self.augment:
            # The disk has no canonical orientation, so all eight dihedral views
            # are exact -- no interpolation and no invented pixels.
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            k = np.random.randint(4)
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            image = np.clip(image * np.random.uniform(0.85, 1.15), 0.0, 1.0)

        stacked = np.stack([np.ascontiguousarray(image), np.ascontiguousarray(mask)])
        return torch.from_numpy(stacked), torch.tensor(float(self.label[index]))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    hits = labels[order].cumsum()
    precision = hits / np.arange(1, len(labels) + 1)
    return float((precision * labels[order]).sum() / labels.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)

    xr.use_spmd()
    devices = xr.global_runtime_device_count()
    device = xm.xla_device()
    if args.batch % devices:
        raise SystemExit(f"batch {args.batch} must divide by {devices}")
    mesh = Mesh(np.arange(devices), (devices, 1, 1, 1), ("data", "c", "h", "w"))

    train_set = CropSet(args.data, "train", True)
    val_set = CropSet(args.data, "val", False)
    rate = train_set.label.mean()
    pos_weight = float((1 - rate) / max(rate, 1e-6))
    print(f"SPMD across {devices} devices | train {len(train_set)} "
          f"({rate:.2%} positive, pos_weight {pos_weight:.2f}) | val {len(val_set)}",
          flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, drop_last=True)

    model = Verifier(args.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0
    since_improvement = 0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        running, batches = torch.zeros((), device=device), 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            xs.mark_sharding(images, mesh, ("data", "c", "h", "w"))
            optimiser.zero_grad()
            loss = positive_weighted_loss(model(images), labels, pos_weight)
            loss.backward()
            optimiser.step()
            xm.mark_step()
            running += loss.detach()
            batches += 1
        scheduler.step()

        model.eval()
        scores, truths = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                xs.mark_sharding(images, mesh, ("data", "c", "h", "w"))
                logits = model(images)
                xm.mark_step()
                scores.append(logits.cpu().numpy())
                truths.append(labels.numpy())
        scores = np.concatenate(scores)
        truths = np.concatenate(truths)
        val_ap = average_precision(scores, truths)

        train_loss = (running / max(batches, 1)).item()
        elapsed = (time.time() - started) / 60
        print(f"epoch {epoch + 1:03d} | train {train_loss:.4f} | "
              f"val AP {val_ap:.4f} | {elapsed:.1f} min", flush=True)

        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({"state": state, "width": args.width}, out_dir / "last.pt")
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save({"state": state, "width": args.width, "val_ap": val_ap},
                       out_dir / "best.pt")
            print(f"  new best AP {best_ap:.4f}", flush=True)
            since_improvement = 0
        else:
            since_improvement += 1

        if since_improvement >= args.patience:
            print(f"early stop: {args.patience} epochs without improvement "
                  f"(best AP {best_ap:.4f})", flush=True)
            break

    print(f"finished. best val AP {best_ap:.4f}", flush=True)
    print(f"raw confidence AP on the same candidates is the bar to clear; "
          f"exp_019 measured it at 0.7150", flush=True)


if __name__ == "__main__":
    main()
