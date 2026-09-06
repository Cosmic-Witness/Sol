"""The refiner network and its loss, importable without an accelerator runtime.

This lives apart from the trainer deliberately. `train_refiner.py` imports
torch_xla at module scope, which exists only on a TPU image — so anything that
merely wants to *use* a trained refiner (inference, evaluation, applying it to a
submission) could not import the class without pulling in a runtime it has no
need for and cannot install. The evaluation kernel failed on exactly that.

Nothing here depends on anything beyond torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
