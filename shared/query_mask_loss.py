"""CPU Hungarian assignment and fixed-shape differentiable instance-mask loss.

Matching is detached and runs on the host. All tensors sent back to XLA retain
the query dimension, including images with fewer targets or no targets.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


@torch.no_grad()
def match_queries(class_logits, mask_logits, targets, active):
    classes = class_logits.detach().float().cpu()
    masks = mask_logits.detach().float().cpu()
    targets = targets.detach().float().cpu()
    active = active.detach().cpu().bool()
    if classes.ndim != 3 or classes.shape[-1] != 2 or masks.ndim != 4:
        raise ValueError('Expected one-class batched query logits')
    batch, queries = classes.shape[:2]
    if masks.shape[:2] != (batch, queries) or targets.shape[:2] != active.shape:
        raise ValueError('Target/query dimensions disagree')
    if targets.shape[0] != batch:
        raise ValueError('Batch dimensions disagree')
    if targets.shape[-2:] != masks.shape[-2:]:
        targets = F.interpolate(targets, size=masks.shape[-2:], mode='nearest')
    if not all(torch.isfinite(value).all() for value in (classes, masks, targets)):
        raise ValueError('Nonfinite matching input')
    labels = torch.ones((batch, queries), dtype=torch.long)
    matched = torch.zeros_like(masks)
    present = torch.zeros((batch, queries), dtype=torch.float32)
    for b in range(batch):
        selected = targets[b, active[b]]
        if len(selected) > queries:
            raise ValueError('More instances than queries')
        if not len(selected):
            continue
        logits = masks[b].flatten(1)
        truth = selected.flatten(1)
        probabilities = logits.sigmoid()
        # Exact dense BCE costs without materializing Q x N x H x W.
        bce = F.softplus(logits).mean(1)[:, None] - logits @ truth.T / logits.shape[1]
        dice = 1 - (2 * probabilities @ truth.T + 1) / (
            probabilities.sum(1)[:, None] + truth.sum(1)[None] + 1)
        cost = -2 * classes[b].softmax(-1)[:, :1] + 5 * bce + 5 * dice
        rows, columns = linear_sum_assignment(cost.numpy())
        labels[b, rows] = 0
        matched[b, rows] = selected[columns]
        present[b, rows] = 1
    return labels, matched, present


def query_mask_loss(class_logits, mask_logits, labels, targets, active):
    """Weighted class CE plus matched BCE/Dice; no variable-length indexing."""
    if class_logits.shape[:2] != labels.shape or mask_logits.shape != targets.shape:
        raise ValueError('Assigned target shapes must equal prediction shapes')
    if active.shape != labels.shape:
        raise ValueError('Active query shape differs from labels')
    class_logp = class_logits.log_softmax(-1)
    class_nll = -class_logp.gather(-1, labels[..., None]).squeeze(-1)
    class_weight = 0.1 + 0.9 * active
    classification = (class_nll * class_weight).sum() / class_weight.sum().clamp_min(1)
    bce_per_query = F.binary_cross_entropy_with_logits(
        mask_logits, targets, reduction='none').mean((-2, -1))
    probability = mask_logits.sigmoid()
    dice_per_query = 1 - (2 * (probability * targets).sum((-2, -1)) + 1) / (
        probability.sum((-2, -1)) + targets.sum((-2, -1)) + 1)
    count = active.sum().clamp_min(1)
    bce = (bce_per_query * active).sum() / count
    dice = (dice_per_query * active).sum() / count
    return 2 * classification + 5 * bce + 5 * dice
