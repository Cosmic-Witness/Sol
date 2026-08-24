"""Semantic probability map -> disjoint filament instances.

This module carries the whole semantic-to-instance step of the baseline, and it
is where the baseline is most likely to lose points. Panoptic Quality punishes
over-fragmentation twice: each spurious piece is a false positive, and the
ground-truth filament it failed to match is a false negative. Splitting one
filament into three costs 1 FN and 3 FP, not a partial credit.

Connected-component labelling is the simplest instance rule that exists. It is
the right baseline precisely because its failure mode is known and measurable,
which gives experiment 002 a clear target.
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.preprocessing import SolarDisk
from shared.utils import IMAGE_SIZE


def probability_to_instances(
    probability: np.ndarray,
    threshold: float,
    min_area: int,
    closing_kernel: int,
    dilate_iterations: int = 0,
    disk: SolarDisk | None = None,
    output_size: tuple[int, int] = IMAGE_SIZE,
) -> list[np.ndarray]:
    """Convert one probability map into a list of disjoint binary masks.

    Steps
    -----
    1. Upsample the probability map to the native 2048 frame before
       thresholding. Thresholding first and upsampling the binary result would
       stair-step every barb.
    2. Threshold.
    3. Morphological closing, which rejoins the fragments of a filament that
       imaging noise broke apart.
    4. Restrict to the solar disk, when the disk geometry is supplied. Nothing
       off-limb can be a filament, and the limb ring produces false positives.
    5. Label connected components and drop anything below `min_area`.

    Components are returned in descending area order. The masks are disjoint by
    construction, because connected components partition the foreground.
    """
    upsampled = cv2.resize(
        probability.astype(np.float32), output_size[::-1], interpolation=cv2.INTER_LINEAR
    )
    binary = (upsampled >= threshold).astype(np.uint8)

    if closing_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if dilate_iterations > 0:
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=dilate_iterations)

    if disk is not None:
        binary &= disk.mask(binary.shape)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    instances: list[tuple[int, np.ndarray]] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        instances.append((area, (labels == label).astype(np.uint8)))

    instances.sort(key=lambda item: -item[0])
    return [mask for _, mask in instances]


def annotations_to_instances(coco, image_id: str) -> list[np.ndarray]:
    """Ground-truth instance masks at native resolution, for local scoring."""
    return [coco.annToMask(ann) for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))]
