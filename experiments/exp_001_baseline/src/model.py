"""Model and loss construction for experiment 001.

The architecture is deliberately conventional. Experiment 001 exists to give a
trustworthy end-to-end pipeline and an honest baseline number, not to win. Every
later experiment is measured against what this produces.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the encoder-decoder named in the config.

    `smp.create_model` accepts the architecture as a string, so switching to
    `unet`, `fpn`, or `deeplabv3plus` in later experiments needs a config edit
    and no code change.
    """
    model_cfg = cfg["model"]
    return smp.create_model(
        arch=model_cfg["arch"],
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    )


class CombinedLoss(nn.Module):
    """Weighted sum of BCE and soft Dice.

    Filament pixels occupy roughly one percent of the on-disk area. Two
    consequences follow, and each term answers one of them:

    - BCE alone reaches a low value by predicting background everywhere, so it
      needs `pos_weight` to make a missed filament pixel expensive.
    - Dice measures overlap rather than per-pixel accuracy, so it keeps
      supplying gradient once the easy background pixels are already correct.
    """

    def __init__(self, bce_weight: float, dice_weight: float, pos_weight: float):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = probs.sum(dims) + targets.sum(dims)
        # The smoothing term keeps an image with no annotated filament from
        # producing a 0/0 gradient.
        dice = 1.0 - ((2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

        return self.bce_weight * bce + self.dice_weight * dice


def build_loss(cfg: dict) -> nn.Module:
    loss_cfg = cfg["loss"]
    return CombinedLoss(
        bce_weight=loss_cfg["bce_weight"],
        dice_weight=loss_cfg["dice_weight"],
        pos_weight=loss_cfg["pos_weight"],
    )
