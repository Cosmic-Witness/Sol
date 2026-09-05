"""Locate the solar disk in a GONG H-alpha frame.

Why this belongs in the pipeline
--------------------------------
A filament is a feature *of the chromosphere*. Any predicted pixel outside the
solar limb is wrong by construction, not merely unlikely — there is no Sun there
to hold a filament. Under Panoptic Quality an unmatched prediction costs half a
unit of denominator, so discarding off-disk predictions is a free reduction in
false positives that requires no model change.

Detection is by intensity rather than by Hough circle transform. The disk is a
single bright region against a near-black sky, so Otsu's threshold separates
them cleanly, and the enclosing circle of the largest component is the limb.
A Hough transform searches a three-parameter space for a shape already trivially
available as a connected component, and is sensitive to radius bounds that vary
with the six GONG stations' different plate scales.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_disk(image: np.ndarray) -> tuple[float, float, float]:
    """Return the disk centre and radius (x, y, r) in pixels.

    Falls back to a centred disk covering most of the frame if segmentation
    fails, so a bad frame degrades to "mask almost nothing" rather than to
    "mask everything" — the safe direction when the output gates predictions.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu on a blurred frame: the sky is near-black, the disk is not, and the
    # blur stops sunspots and filaments from fragmenting the disk component.
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        height, width = image.shape
        return width / 2.0, height / 2.0, min(height, width) * 0.49

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[largest, cv2.CC_STAT_AREA])

    # The disk fills roughly a fifth to a half of the frame. A component far
    # outside that is a cloud bank or an artefact, not the Sun.
    height, width = image.shape
    if not (0.05 * height * width < area < 0.90 * height * width):
        return width / 2.0, height / 2.0, min(height, width) * 0.49

    x = float(centroids[largest][0])
    y = float(centroids[largest][1])
    # Equivalent-area radius is far more robust than the enclosing circle, which
    # a single bright cloud touching the limb would inflate.
    radius = float(np.sqrt(area / np.pi))
    return x, y, radius


def disk_mask(shape: tuple[int, int], centre_x: float, centre_y: float,
              radius: float, margin: float = 1.02) -> np.ndarray:
    """Binary mask of the disk, dilated slightly.

    The margin exists because annotators trace filaments that touch the limb and
    the traced boundary can sit a pixel or two outside the fitted circle.
    Trimming those would convert correct predictions into partial ones, which is
    the opposite of the intent.
    """
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (((xx - centre_x) ** 2 + (yy - centre_y) ** 2) <= (radius * margin) ** 2).astype(np.uint8)


def mask_to_disk(binary: np.ndarray, disk: np.ndarray) -> np.ndarray:
    """Restrict a prediction to the disk."""
    return (binary.astype(np.uint8) & disk).astype(np.uint8)
