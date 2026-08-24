"""Tests for the metric and submission helpers in `shared/utils.py`.

The Panoptic Quality implementation decides which checkpoint gets submitted, so
a silent error here would misdirect every later experiment. These cases pin the
behaviour at the boundaries the metric definition cares about.

    pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.utils import (
    aggregate_pq,
    check_no_overlap,
    compute_pq,
    decode_rle,
    encode_rle,
    paint_panoptic,
    write_submission,
)

SIZE = (256, 256)


def box(y0: int, y1: int, x0: int, x1: int, size=SIZE) -> np.ndarray:
    mask = np.zeros(size, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


# --------------------------------------------------------------------------- #
# RLE
# --------------------------------------------------------------------------- #
def test_rle_round_trip():
    original = box(10, 40, 20, 90)
    assert np.array_equal(decode_rle(encode_rle(original), SIZE), original)


def test_rle_handles_empty_mask():
    empty = np.zeros(SIZE, dtype=np.uint8)
    assert decode_rle(encode_rle(empty), SIZE).sum() == 0


# --------------------------------------------------------------------------- #
# Panoptic Quality
# --------------------------------------------------------------------------- #
def test_perfect_prediction_scores_one():
    masks = [box(0, 50, 0, 50), box(100, 160, 100, 160)]
    result = compute_pq(masks, masks)
    assert result["pq"] == pytest.approx(1.0)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)


def test_empty_prediction_scores_zero():
    result = compute_pq([], [box(0, 50, 0, 50)])
    assert result["pq"] == 0.0
    assert result["fn"] == 1


def test_iou_below_threshold_is_not_a_match():
    """A 0.5 IoU must not match: the definition requires strictly greater."""
    truth = box(0, 100, 0, 100)
    # Half the area, fully contained -> IoU exactly 0.5.
    prediction = box(0, 50, 0, 100)
    result = compute_pq([prediction], [truth])
    assert result["tp"] == 0
    assert result["fp"] == 1 and result["fn"] == 1
    assert result["pq"] == 0.0


def test_fragmentation_is_charged_twice():
    """One truth split into three pieces costs 3 FP and 1 FN, not partial credit."""
    truth = box(0, 90, 0, 30)
    pieces = [box(0, 30, 0, 30), box(30, 60, 0, 30), box(60, 90, 0, 30)]
    result = compute_pq(pieces, [truth])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 3, 1)


def test_one_prediction_cannot_match_two_truths():
    truth = [box(0, 50, 0, 50), box(0, 50, 60, 110)]
    prediction = [box(0, 50, 0, 110)]
    result = compute_pq(prediction, truth)
    assert result["tp"] <= 1


def test_aggregate_pools_counts_rather_than_averaging():
    """Pooling must weight a 20-filament image above a 1-filament image."""
    a = compute_pq([box(0, 50, 0, 50)], [box(0, 50, 0, 50)])          # perfect, 1 instance
    b = compute_pq([], [box(0, 50, 0, 50), box(60, 100, 60, 100)])     # missed, 2 instances
    pooled = aggregate_pq([a, b])
    assert pooled["tp"] == 1 and pooled["fn"] == 2
    assert pooled["pq"] == pytest.approx(1 / (1 + 0.5 * 2))


# --------------------------------------------------------------------------- #
# Overlap handling
# --------------------------------------------------------------------------- #
def test_paint_panoptic_returns_disjoint_masks():
    high = box(0, 100, 0, 100)
    low = box(50, 150, 50, 150)  # overlaps `high` on a 50x50 square
    kept = paint_panoptic([(0.9, high), (0.4, low)], min_area=10, image_size=SIZE)
    assert len(kept) == 2
    total = sum(mask for _, mask, _ in kept)
    assert total.max() <= 1, "painted masks must not share a pixel"
    assert kept[0][2] == 100 * 100
    assert kept[1][2] == 100 * 100 - 50 * 50


def test_paint_panoptic_drops_slivers_below_min_area():
    big = box(0, 100, 0, 100)
    sliver = box(99, 100, 0, 100)  # entirely swallowed except nothing
    kept = paint_panoptic([(0.9, big), (0.5, sliver)], min_area=10, image_size=SIZE)
    assert len(kept) == 1


def test_write_submission_and_validator_agree(tmp_path):
    csv_path = tmp_path / "submission.csv"
    predictions = {
        "20150125172714Mh": [box(0, 50, 0, 50, SIZE), box(60, 110, 60, 110, SIZE)],
        "20150126172714Mh": [box(0, 50, 0, 50, SIZE)],
    }
    frame = write_submission(predictions, str(csv_path))
    assert list(frame["filament_id"]) == [
        "20150125172714Mh_1",
        "20150125172714Mh_2",
        "20150126172714Mh_1",
    ]
    check_no_overlap(str(csv_path), SIZE)


def test_validator_rejects_overlapping_submission(tmp_path):
    csv_path = tmp_path / "bad.csv"
    overlapping = [box(0, 50, 0, 50, SIZE), box(25, 75, 25, 75, SIZE)]
    write_submission({"20150125172714Mh": overlapping}, str(csv_path))
    with pytest.raises(ValueError, match="Overlapping"):
        check_no_overlap(str(csv_path), SIZE)
