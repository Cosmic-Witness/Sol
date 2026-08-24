"""Training entry point for experiment 001.

Run from the repository root:

    python -m experiments.exp_001_baseline.src.train \
        --config experiments/exp_001_baseline/config.yaml

Colab survival
--------------
Free-tier Colab reclaims a runtime without warning, and it does so on Google's
schedule rather than yours. Every design choice below follows from that:

- A checkpoint is written after every epoch, holding the model, the optimizer,
  the schedule position, the AMP scaler, and the RNG states. Weights alone are
  not enough; restoring them without the optimizer restarts the momentum and the
  learning-rate schedule, which wastes several epochs of progress.
- `--resume` is the default. Re-running the identical command after a disconnect
  continues from the last completed epoch.
- The checkpoint directory belongs on Google Drive, not on the runtime disk. The
  runtime disk disappears with the runtime.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_001_baseline.src.dataset import FilamentDataset  # noqa: E402
from experiments.exp_001_baseline.src.model import build_loss, build_model  # noqa: E402
from experiments.exp_001_baseline.src.evaluate import evaluate_pq  # noqa: E402
from shared.data_split import assert_disjoint, load_records, make_split  # noqa: E402
from shared.utils import set_seed  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Cosine decay with a short linear warmup.

    The warmup matters because the decoder starts from random weights while the
    encoder starts from ImageNet. Without it, the first few large steps damage
    the pretrained features the experiment depends on.
    """
    epochs = cfg["train"]["epochs"]
    warmup_steps = max(1, cfg["train"]["warmup_epochs"] * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, **state) -> None:
    """Write atomically.

    A runtime killed mid-write would otherwise leave a truncated file, and the
    next resume would fail on the very checkpoint meant to protect it.
    """
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def run_validation(model, loader, criterion, device) -> tuple[float, float]:
    """Return mean validation loss and mean Dice at training resolution.

    This is the cheap per-epoch signal. Panoptic Quality is the metric that
    decides the experiment, but it costs a full-resolution instance decode, so
    it runs on a subset. See `evaluate_pq`.
    """
    model.eval()
    losses, dices = [], []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=images.is_cuda):
                logits = model(images)
                loss = criterion(logits, masks)
            probs = torch.sigmoid(logits.float())
            pred = (probs >= 0.5).float()
            intersection = (pred * masks).sum((1, 2, 3))
            cardinality = pred.sum((1, 2, 3)) + masks.sum((1, 2, 3))
            dices.append(((2 * intersection + 1) / (cardinality + 1)).mean().item())
            losses.append(loss.item())
    return float(np.mean(losses)), float(np.mean(dices))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the experiment 001 baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and train from scratch.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    paths = cfg["paths"]
    ckpt_dir = Path(paths["checkpoint_dir"])
    log_dir = Path(paths["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    metrics_csv = log_dir / "metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: no GPU visible. Training on CPU is not viable here.", flush=True)

    split = make_split(paths["annotations"], cfg["data"]["val_fraction"], cfg["seed"])
    assert_disjoint(split)
    _, id_to_stem = load_records(paths["annotations"])
    print(split.summary(), flush=True)

    common = dict(
        id_to_stem=id_to_stem,
        cache_dir=paths["cache_dir"],
        image_size=cfg["data"]["image_size"],
    )
    train_loader = DataLoader(
        FilamentDataset(split.train_image_ids, train=True, **common),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        FilamentDataset(split.val_image_ids, train=False, **common),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    start_epoch = 0
    best_pq = 0.0
    epochs_without_gain = 0

    if last_ckpt.exists() and not args.no_resume:
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_pq = state["best_pq"]
        epochs_without_gain = state.get("epochs_without_gain", 0)
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        print(f"resumed from epoch {state['epoch']}, best PQ {best_pq:.4f}", flush=True)

    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["epoch", "train_loss", "val_loss", "val_dice", "val_pq", "val_sq", "val_rq", "lr", "seconds"]
            )

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        started = time.time()
        model.train()
        running = []
        optimizer.zero_grad(set_to_none=True)

        for step, (images, masks, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(images), masks) / cfg["train"]["grad_accum_steps"]
            scaler.scale(loss).backward()

            if (step + 1) % cfg["train"]["grad_accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running.append(loss.item() * cfg["train"]["grad_accum_steps"])

        train_loss = float(np.mean(running))
        val_loss, val_dice = run_validation(model, val_loader, criterion, device)
        pq = evaluate_pq(model, cfg, split.val_image_ids, id_to_stem, device, subset=cfg["train"]["pq_subset"])
        elapsed = time.time() - started

        print(
            f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"dice {val_dice:.4f} | PQ {pq['pq']:.4f} (SQ {pq['sq']:.3f} RQ {pq['rq']:.3f}) | {elapsed:.0f}s",
            flush=True,
        )
        with open(metrics_csv, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [epoch, train_loss, val_loss, val_dice, pq["pq"], pq["sq"], pq["rq"],
                 optimizer.param_groups[0]["lr"], round(elapsed, 1)]
            )

        checkpoint = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            best_pq=max(best_pq, pq["pq"]),
            epochs_without_gain=epochs_without_gain,
            config=cfg,
            rng_python=random.getstate(),
            rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(),
        )
        save_checkpoint(last_ckpt, **checkpoint)

        if pq["pq"] > best_pq:
            best_pq = pq["pq"]
            epochs_without_gain = 0
            checkpoint["best_pq"] = best_pq
            save_checkpoint(best_ckpt, **checkpoint)
            print(f"  new best PQ {best_pq:.4f}", flush=True)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg["train"]["early_stopping_patience"]:
                print(f"early stop: {epochs_without_gain} epochs without gain", flush=True)
                break

    print(f"finished. best validation PQ {best_pq:.4f}", flush=True)


if __name__ == "__main__":
    main()
