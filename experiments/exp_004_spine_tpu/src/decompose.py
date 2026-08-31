"""Turn a filament mask and a spine map into instances.

Connected-component labelling answers "which pixels touch each other". The task
needs "which pixels belong to the same filament", and those differ precisely
when the sky breaks a filament into pieces — the case that cost exp_001 most of
its recognition quality.

Seeding on spines answers the right question. Each filament contributes one
spine, so the number of spine components is the number of filaments, and every
mask pixel is then assigned to the spine it is closest to *through the mask*.
Fragments of one filament all lie nearest the same spine and are reunited;
genuinely distinct filaments keep distinct spines and stay apart.
"""

from __future__ import annotations

import cv2
import numpy as np


def _nearest_seed_labels(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Assign every mask pixel the label of the nearest seed.

    `cv2.distanceTransformWithLabels` computes, for each zero pixel, the distance
    to the nearest non-zero pixel and the label of that pixel, in one pass. Using
    it on the inverted seed image gives a Voronoi partition keyed by seed label
    far faster than iterating components.
    """
    background = (seeds == 0).astype(np.uint8)
    _, index = cv2.distanceTransformWithLabels(
        background, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # index holds, per pixel, the index of its nearest zero-of-background (i.e.
    # nearest seed pixel). Map those back to seed component labels.
    flat_seed_labels = seeds.reshape(-1)
    nonzero_positions = np.flatnonzero(flat_seed_labels)
    lookup = np.zeros(nonzero_positions.size + 1, dtype=np.int32)
    lookup[1:] = flat_seed_labels[nonzero_positions]
    assigned = lookup[index]
    return np.where(mask > 0, assigned, 0)


def decompose(
    mask: np.ndarray,
    spine: np.ndarray,
    min_area: int = 150,
    min_spine_area: int = 8,
) -> list[np.ndarray]:
    """Split a binary filament mask into instances using the spine map.

    Falls back to plain connected components when no spine survives, so a failure
    of the spine head degrades to exp_001's behaviour rather than to nothing.
    """
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return []

    spine = ((spine > 0) & (mask > 0)).astype(np.uint8)
    # A spine only counts if it survives inside the predicted mask; a few stray
    # pixels are noise, not a filament.
    n_spine, spine_labels, stats, _ = cv2.connectedComponentsWithStats(spine, connectivity=8)
    keep = [i for i in range(1, n_spine) if stats[i, cv2.CC_STAT_AREA] >= min_spine_area]

    if not keep:
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        return [
            m for m in ((labels == i).astype(np.uint8) for i in range(1, n))
            if int(m.sum()) >= min_area
        ]

    seeds = np.zeros_like(spine_labels, dtype=np.int32)
    for new_label, old_label in enumerate(keep, start=1):
        seeds[spine_labels == old_label] = new_label

    assigned = _nearest_seed_labels(mask, seeds)

    instances = []
    for label in range(1, len(keep) + 1):
        instance = (assigned == label).astype(np.uint8)
        if int(instance.sum()) >= min_area:
            instances.append(instance)
    return instances


def connected_components(mask: np.ndarray, min_area: int = 150) -> list[np.ndarray]:
    """exp_001's decomposition, kept here so the two can be compared directly."""
    mask = (mask > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    return [
        m for m in ((labels == i).astype(np.uint8) for i in range(1, n))
        if int(m.sum()) >= min_area
    ]
