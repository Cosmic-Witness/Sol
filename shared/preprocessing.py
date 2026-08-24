"""H-alpha image conditioning for GONG full-disk observations.

The pipeline follows the sequence established in the EdgeAttNet work
(arXiv:2509.02964) and the competition discussion:

    1. detect the solar disk
    2. mask everything outside the limb
    3. flatten the radial limb-darkening gradient
    4. smooth lightly to suppress sensor and seeing noise
    5. equalise local contrast with CLAHE

Step 3 is the one that matters most. Untreated, the disk centre is markedly
brighter than the limb, so a single global threshold or a shallow feature map
sees "dark" very differently at the centre than near the edge. Flattening makes
filament contrast comparable across the whole disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SolarDisk:
    """Detected solar disk geometry, in pixel coordinates."""

    cx: float
    cy: float
    radius: float

    def mask(self, shape: tuple[int, int], shrink: float = 0.99) -> np.ndarray:
        """Binary mask of the on-disk region.

        `shrink` pulls the edge slightly inside the limb. The outermost ring
        carries strong intensity gradients and compression artefacts, and it
        produces spurious dark detections if left in.
        """
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        r2 = (xx - self.cx) ** 2 + (yy - self.cy) ** 2
        return (r2 <= (self.radius * shrink) ** 2).astype(np.uint8)


def detect_disk(image: np.ndarray) -> SolarDisk:
    """Locate the solar disk in a grayscale H-alpha frame.

    A Hough circle transform is tried first. It is precise when the limb is
    clean. Cloud cover and poor seeing can defeat it, so a threshold-and-moments
    fallback runs when Hough returns nothing. The fallback is robust because the
    disk is by far the brightest connected region in the frame.
    """
    h, w = image.shape[:2]
    blurred = cv2.medianBlur(image, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=float(min(h, w)),
        param1=100,
        param2=50,
        minRadius=int(0.30 * min(h, w)),
        maxRadius=int(0.50 * min(h, w)),
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        return SolarDisk(float(cx), float(cy), float(r))

    # Fallback: Otsu threshold, then take the largest component.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 1:
        # Degenerate frame. Assume a centred disk covering most of the image.
        return SolarDisk(w / 2.0, h / 2.0, 0.45 * min(h, w))
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[largest]
    area = float(stats[largest, cv2.CC_STAT_AREA])
    return SolarDisk(float(cx), float(cy), float(np.sqrt(area / np.pi)))


def flatten_limb_darkening(
    image: np.ndarray, disk: SolarDisk, n_bins: int = 64
) -> np.ndarray:
    """Remove the radial intensity gradient across the disk.

    The median intensity is measured in `n_bins` annuli of equal radial width,
    then each pixel is divided by the interpolated profile value at its radius.
    The median resists filaments and sunspots, so dark features do not drag the
    profile down and erase themselves.

    Returns a float32 array centred near 1.0 on quiet disk.
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    radius = np.sqrt((xx - disk.cx) ** 2 + (yy - disk.cy) ** 2) / disk.radius

    on_disk = radius <= 0.99
    img = image.astype(np.float32)

    edges = np.linspace(0.0, 0.99, n_bins + 1)
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ring = on_disk & (radius >= lo) & (radius < hi)
        if ring.sum() < 64:
            continue
        centres.append((lo + hi) / 2.0)
        medians.append(float(np.median(img[ring])))
    if len(centres) < 2:
        return img

    profile = np.interp(radius, centres, medians, left=medians[0], right=medians[-1])
    profile = np.maximum(profile, 1e-3)
    flattened = img / profile
    return np.where(on_disk, flattened, 0.0).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Local contrast equalisation on an 8-bit image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(image)


def preprocess(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    blur_sigma: float = 0.7,
    return_disk: bool = False,
):
    """Run the full conditioning pipeline on one grayscale frame.

    Parameters
    ----------
    image
        uint8 array of shape (H, W). Grayscale. Never pass an RGB frame.

    Returns
    -------
    uint8 array of shape (H, W), off-disk pixels set to 0. When `return_disk`
    is true, also returns the detected `SolarDisk`, which the instance
    post-processing needs to reject off-limb detections.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale frame, received shape {image.shape}")

    disk = detect_disk(image)
    flattened = flatten_limb_darkening(image, disk)

    # Rescale the flattened field into 8-bit using robust percentiles, so a few
    # saturated plage pixels cannot compress the filament contrast range.
    on_disk = disk.mask(image.shape).astype(bool)
    values = flattened[on_disk]
    lo, hi = np.percentile(values, [0.5, 99.5])
    scaled = np.clip((flattened - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    scaled = (scaled * 255.0).astype(np.uint8)

    if blur_sigma > 0:
        scaled = cv2.GaussianBlur(scaled, (3, 3), blur_sigma)
    scaled = apply_clahe(scaled, clahe_clip, clahe_grid)
    scaled[~on_disk] = 0

    if return_disk:
        return scaled, disk
    return scaled
