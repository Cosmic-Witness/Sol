"""Differential test: our PQ against the organiser's published implementation.

The competition's self-evaluation notebook (azimahmadzadeh/self-evaluation-notebook)
defines the leaderboard metric. Its scoring differs from ours in form: it marks
every GT/prediction pair above the IoU threshold as a true positive, while
`shared.utils.compute_pq` performs a greedy one-to-one match. If those disagree,
every threshold and resolution decision taken on validation is measuring the
wrong thing.

They are equivalent exactly when masks are pixel-disjoint, and both sides of the
comparison are: the competition rejects submissions whose predictions overlap,
and the ground truth is disjoint by construction (verified: 0 of 107 sampled
images contain overlapping GT instances).

The reason is a counting argument. If two disjoint predictions P1 and P2 both
exceeded IoU 0.5 against one ground-truth mask G, then each intersection would
have to exceed half of |G|, and their sum would exceed |G| — impossible for
disjoint sets. The same argument bounds one prediction against two disjoint GTs.
Above a threshold of 0.5 the match is therefore unique, and "all qualifying
pairs" and "one-to-one matching" coincide.

This test pins that equivalence on disjoint inputs, and documents the divergence
on overlapping ones so the assumption is visible rather than implicit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shared.utils import aggregate_pq, compute_pq

SIZE = 96


def official_pq(per_image):
    """Faithful reimplementation of get_pq_score from the organiser's notebook."""
    tp_scores, fp, fn = [], 0, 0
    for gts, preds in per_image:
        n_gt, n_pred = len(gts), len(preds)
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        g = torch.tensor(np.stack(gts).reshape(n_gt, -1), dtype=torch.float32)
        p = torch.tensor(np.stack(preds).reshape(n_pred, -1), dtype=torch.float32)
        intersection = g @ p.t()
        union = g.sum(1).view(-1, 1) + p.sum(1).view(1, -1) - intersection
        iou = torch.where(union == 0, torch.tensor(0.0), intersection / union)
        hit = iou > 0.5
        tp_scores.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    denominator = len(tp_scores) + 0.5 * fp + 0.5 * fn
    return (sum(tp_scores) / denominator if denominator > 0 else 0.0), len(tp_scores), fp, fn


def disjoint_masks(rng, count):
    """Random rectangles that never overlap, as the scorer requires."""
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    out = []
    for _ in range(count):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        y, x = rng.integers(2, SIZE - 24, 2)
        h, w = rng.integers(8, 22, 2)
        mask[y:y + h, x:x + w] = 1
        mask = (mask & ~canvas).astype(np.uint8)
        if mask.sum() > 16:
            out.append(mask)
            canvas |= mask
    return out


@pytest.mark.parametrize("seed", range(25))
def test_matches_official_on_disjoint_masks(seed):
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(rng.integers(1, 4)):
        images.append((disjoint_masks(rng, rng.integers(0, 5)),
                       disjoint_masks(rng, rng.integers(0, 5))))

    expected_pq, tp, fp, fn = official_pq(images)
    ours = aggregate_pq([compute_pq(preds, gts) for gts, preds in images])

    # Counts must agree exactly; they are integers and carry the whole argument.
    assert (ours["tp"], ours["fp"], ours["fn"]) == (tp, fp, fn)
    # PQ agrees only to float32: the organiser accumulates IoU in torch.float32
    # while pycocotools returns float64, so the two differ in the eighth decimal
    # on identical inputs. A tolerance below that measures the dtype, not the
    # metric.
    assert ours["pq"] == pytest.approx(expected_pq, abs=1e-6)


def test_overlapping_predictions_are_where_the_two_diverge():
    """Documents the assumption rather than asserting agreement.

    Two identical predictions against one ground truth: the organiser's rule
    counts both as true positives, ours counts one and calls the other a false
    positive. Such a submission is rejected by the competition, so the case
    cannot arise on the leaderboard — but if that rule ever changed, this test
    is where the divergence would surface.
    """
    gt = np.zeros((SIZE, SIZE), np.uint8)
    gt[10:40, 10:40] = 1
    duplicated = [gt.copy(), gt.copy()]

    official_tp = official_pq([([gt], duplicated)])[1]
    ours = compute_pq(duplicated, [gt])

    assert official_tp == 2
    assert ours["tp"] == 1 and ours["fp"] == 1
