"""Tests for the COCO to YOLO segmentation conversion.

A defect here is expensive in a way a defect elsewhere is not: it corrupts the
labels silently, training runs for hours on them, and the result looks like a
modelling failure rather than a data one. These cases pin the properties the
converter must preserve.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp_002_yolo_seg.src.prepare_yolo import merge_multi_segment, write_label


def square(x0: float, y0: float, side: float) -> list[float]:
    """A COCO-style flat polygon [x1, y1, x2, y2, ...]."""
    return [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side]


# --------------------------------------------------------------------------- #
# merge_multi_segment
# --------------------------------------------------------------------------- #
def test_single_ring_is_returned_unchanged():
    ring = square(0, 0, 10)
    merged = merge_multi_segment([ring])
    assert np.array_equal(merged, np.array(ring, dtype=np.float64).reshape(-1, 2))


def test_merge_keeps_every_vertex_of_every_ring():
    """No fragment of a filament may be dropped by the stitching."""
    rings = [square(0, 0, 10), square(100, 100, 10)]
    merged = merge_multi_segment(rings)

    for ring in rings:
        for point in np.array(ring).reshape(-1, 2):
            assert np.isclose(merged, point).all(axis=1).any(), f"lost vertex {point}"


def test_merge_bridges_at_the_closest_pair():
    """The seam should run along the shortest bridge, not an arbitrary one.

    Two squares are placed so that one specific corner pair is much closer than
    any other. Walking the merged path, consecutive vertices drawn from
    different rings must be that pair.
    """
    left = square(0, 0, 10)          # occupies x in [0, 10]
    right = square(12, 0, 10)        # occupies x in [12, 22]; gap of 2 at y=0 and y=10
    merged = merge_multi_segment([left, right])

    left_points = {tuple(p) for p in np.array(left).reshape(-1, 2)}
    crossings = []
    for a, b in zip(merged[:-1], merged[1:]):
        if (tuple(a) in left_points) != (tuple(b) in left_points):
            crossings.append(float(np.linalg.norm(a - b)))

    assert crossings, "the merged path never crosses between rings"
    # The closest approach between the two squares is exactly 2.0.
    assert min(crossings) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# write_label
# --------------------------------------------------------------------------- #
def test_label_is_normalised_and_recoverable(tmp_path):
    polygon = np.array(square(256, 512, 1024), dtype=np.float64).reshape(-1, 2)
    path = tmp_path / "label.txt"

    assert write_label(path, [polygon], 2048, 2048) == 1

    fields = path.read_text().strip().split()
    assert fields[0] == "0", "the dataset has exactly one class"

    values = np.array([float(v) for v in fields[1:]], dtype=np.float64)
    assert len(values) % 2 == 0
    assert ((values >= 0.0) & (values <= 1.0)).all(), "coordinates must be normalised"
    assert np.allclose(values.reshape(-1, 2) * 2048, polygon, atol=1e-2)


def test_vertices_outside_the_frame_are_clipped_not_dropped(tmp_path):
    """A vertex a pixel outside the frame is an annotation rounding artefact.

    Dropping it would lose the filament; clipping keeps the instance and costs a
    pixel of boundary.
    """
    polygon = np.array([[-5.0, -5.0], [2100.0, 10.0], [1000.0, 2100.0]])
    path = tmp_path / "label.txt"

    assert write_label(path, [polygon], 2048, 2048) == 1

    values = np.array([float(v) for v in path.read_text().split()[1:]])
    assert ((values >= 0.0) & (values <= 1.0)).all()
    assert len(values) == 6, "all three vertices survive"


def test_degenerate_polygon_is_skipped(tmp_path):
    """Fewer than three vertices is not a region and cannot be an instance."""
    path = tmp_path / "label.txt"
    assert write_label(path, [np.array([[1.0, 1.0], [2.0, 2.0]])], 2048, 2048) == 0
    assert path.read_text() == ""
