"""Group foreground pixels into instances using predicted centers and offsets.

Each foreground pixel predicts a vector to the center of the filament it belongs
to. Voting those predictions against a set of detected centers assigns every
pixel an instance, so grouping is decided by what the network learned rather than
by whether the pixels happen to touch. Two disconnected patches of one filament
point at the same center and are reunited; two touching filaments point at
different centers and are separated. Connected components can do neither.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_centers(heatmap: np.ndarray, threshold: float = 0.30, nms_kernel: int = 21) -> np.ndarray:
    """Peak-pick the center heatmap. Returns an (N, 2) array of (y, x)."""
    # Greyscale dilation keeps, at every pixel, the maximum over its
    # neighbourhood; a pixel equal to that maximum is a local peak. This is
    # max-pool NMS without a framework.
    dilated = cv2.dilate(heatmap, np.ones((nms_kernel, nms_kernel), np.uint8))
    peaks = (heatmap >= dilated) & (heatmap >= threshold)
    ys, xs = np.nonzero(peaks)
    return np.stack([ys, xs], axis=1) if ys.size else np.empty((0, 2), dtype=np.int64)


def group(
    mask: np.ndarray,
    heatmap: np.ndarray,
    offsets: np.ndarray,
    threshold: float = 0.30,
    nms_kernel: int = 21,
    min_area: int = 150,
) -> list[np.ndarray]:
    """Return per-instance binary masks.

    `offsets` is (2, H, W) holding the vector (dy, dx) from each pixel to its
    filament's center, in pixels.
    """
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return []

    centers = find_centers(heatmap, threshold, nms_kernel)
    if centers.shape[0] == 0:
        # No center survived: fall back to connectivity rather than returning
        # nothing, so a weak center head degrades to exp_001 behaviour.
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        return [m for m in ((labels == i).astype(np.uint8) for i in range(1, n))
                if int(m.sum()) >= min_area]

    ys, xs = np.nonzero(mask)
    # Where each foreground pixel votes its filament's center to be.
    voted = np.stack([ys + offsets[0][ys, xs], xs + offsets[1][ys, xs]], axis=1)

    # Nearest center to each vote. Chunked so a full-disk mask does not build a
    # (foreground x centers) matrix in one allocation.
    assignment = np.empty(voted.shape[0], dtype=np.int32)
    step = 200_000
    for start in range(0, voted.shape[0], step):
        chunk = voted[start:start + step]
        distances = np.linalg.norm(chunk[:, None, :] - centers[None, :, :].astype(np.float32), axis=2)
        assignment[start:start + step] = np.argmin(distances, axis=1)

    instances = []
    for index in range(centers.shape[0]):
        selected = assignment == index
        if not selected.any():
            continue
        instance = np.zeros_like(mask)
        instance[ys[selected], xs[selected]] = 1
        if int(instance.sum()) >= min_area:
            instances.append(instance)
    return instances


def targets_from_labels(labels: np.ndarray, sigma: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Build the center heatmap and offset field from an instance-label map."""
    size = labels.shape[0]
    heatmap = np.zeros((size, size), dtype=np.float32)
    offsets = np.zeros((2, size, size), dtype=np.float32)

    for index in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == index)
        if ys.size == 0:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        offsets[0][ys, xs] = cy - ys
        offsets[1][ys, xs] = cx - xs

        # An isotropic Gaussian bump at the centroid; peaks are what inference
        # looks for, so the target is what a peak-picker should see.
        y0, y1 = max(0, int(cy) - 3 * int(sigma)), min(size, int(cy) + 3 * int(sigma) + 1)
        x0, x1 = max(0, int(cx) - 3 * int(sigma)), min(size, int(cx) + 3 * int(sigma) + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        np.maximum(heatmap[y0:y1, x0:x1], bump, out=heatmap[y0:y1, x0:x1])

    return heatmap, offsets
