"""The verifier: is there really a filament under this proposed mask?

Kept deliberately small. The task is a binary judgement on a 128-pixel crop with
a few thousand training examples, and the two networks this project has trained
at greater capacity both overfit their training set while validation stood still.

Two input channels, the photograph and the proposed mask. The mask channel is
what makes the question well posed: several filaments can share a crop, and
without it the model cannot tell which one it is being asked about.

Lives in its own module because the trainer imports `torch_xla` at module scope,
which is absent everywhere except a TPU kernel. Every consumer of this file must
be importable without it -- a mistake this project has already made twice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Verifier(nn.Module):
    """128 -> 64 -> 32 -> 16 -> 8 -> 4, then pooled to a single logit."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            block(2, w), block(w, w * 2), block(w * 2, w * 4),
            block(w * 4, w * 4), block(w * 4, w * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(w * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def positive_weighted_loss(logits: torch.Tensor, labels: torch.Tensor,
                           pos_weight: float) -> torch.Tensor:
    """Plain BCE, with the positives upweighted to offset a ~10% base rate.

    Left as logits rather than probabilities because the emission rule sweeps a
    threshold afterwards, and a calibrated ordering matters more than a
    calibrated absolute value.
    """
    weight = torch.where(labels > 0.5,
                         torch.full_like(labels, pos_weight),
                         torch.ones_like(labels))
    return nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=weight)
