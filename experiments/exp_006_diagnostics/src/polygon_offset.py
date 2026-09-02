"""Find the inward polygon offset that makes training targets match the scorer.

The problem this solves
-----------------------
Ultralytics rasterises training targets with `cv2.fillPoly`; the competition
scorer rasterises with `pycocotools`. Measured on this dataset the two disagree
at IoU 0.9016, with the training target 10.8% larger in area. The model is
therefore trained to reproduce masks that are fat relative to what the
leaderboard measures.

Correcting it after the fact does not work cleanly. One pixel of erosion is the
finest operation available on a raster — thresholding the distance transform
below 1.0 returns the mask unchanged, because boundary pixels sit at distance
exactly 1.0 — and one pixel overshoots, taking the area from 11% too large to
10% too small.

In *coordinate* space there is no such floor. Shrinking the polygon inward by a
fraction of a pixel before rasterisation is well defined, and it is applied
before Ultralytics ever sees the label, so it needs no patch to the library.

The offset is applied per ring with Shapely's buffer, which handles the cases a
naive centroid-scaling would get wrong: a filament is long and curved, so
scaling toward its centroid would move the ends far more than the middle, and a
buffer moves every edge by the same perpendicular distance, which is what a
rasterisation boundary offset actually is.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pycocotools.mask as mask_util
from shapely.geometry import Polygon

FULL = 2048


def shrink_ring(ring: list[float], offset: float) -> list[list[float]]:
    """Offset one polygon ring inward by `offset` pixels.

    A buffer can split a thin shape into several pieces or erase it entirely;
    both are returned faithfully rather than papered over, since an offset large
    enough to destroy a filament is information about the offset.
    """
    points = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return [ring]
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)          # repairs self-intersecting traces
    if polygon.is_empty:
        return [ring]

    shrunk = polygon.buffer(-offset)
    if shrunk.is_empty:
        return []
    geoms = [shrunk] if shrunk.geom_type == "Polygon" else list(shrunk.geoms)
    out = []
    for geom in geoms:
        coords = np.asarray(geom.exterior.coords, dtype=np.float64)
        if len(coords) >= 3:
            out.append(coords.reshape(-1).tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--sample", type=int, default=250)
    args = parser.parse_args()

    from ultralytics.data.utils import polygon2mask

    with open(args.annotations, encoding="utf-8") as fh:
        coco = json.load(fh)
    annotations = [a for a in coco["annotations"]
                   if isinstance(a.get("segmentation"), list) and a["segmentation"]]
    rng = np.random.default_rng(2026)
    sample = rng.choice(len(annotations), size=min(args.sample, len(annotations)), replace=False)

    def as_ultralytics(rings: list[list[float]]) -> np.ndarray:
        """Rasterise exactly as Ultralytics does, one ring at a time.

        `polygon2mask` calls np.asarray on its whole polygon list, so it requires
        every ring to carry the same vertex count. Shrinking changes vertex
        counts per ring, so each is rasterised separately and combined — which is
        equivalent, since fillPoly over a list is a union anyway.
        """
        mask = np.zeros((FULL, FULL), np.uint8)
        for ring in rings:
            poly = np.asarray(ring, np.float32).reshape(-1, 2)
            if len(poly) < 3:
                continue
            mask |= (polygon2mask((FULL, FULL), [poly]) > 0).astype(np.uint8)
        return mask

    offsets = (0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0)
    stats = {o: {"iou": [], "ratio": [], "lost": 0} for o in offsets}

    for index in sample:
        rings = [r for r in annotations[index]["segmentation"] if len(r) >= 6]
        if not rings:
            continue
        scorer = mask_util.decode(
            mask_util.merge(mask_util.frPyObjects(rings, FULL, FULL))).astype(np.uint8)
        if scorer.sum() < 100:
            continue

        for offset in offsets:
            if offset == 0.0:
                shrunk = rings
            else:
                shrunk = [s for r in rings for s in shrink_ring(r, offset)]
            if not shrunk:
                stats[offset]["lost"] += 1
                continue
            target = as_ultralytics(shrunk)
            if target.sum() == 0:
                stats[offset]["lost"] += 1
                continue
            intersection = int((target & scorer).sum())
            union = int((target | scorer).sum())
            stats[offset]["iou"].append(intersection / max(union, 1))
            stats[offset]["ratio"].append(target.sum() / scorer.sum())

    print(f"{'offset px':>10}{'mean IoU':>11}{'median IoU':>12}{'area ratio':>12}{'lost':>7}")
    for offset in offsets:
        s = stats[offset]
        if not s["iou"]:
            print(f"{offset:>10.2f}{'—':>11}{'—':>12}{'—':>12}{s['lost']:>7}")
            continue
        print(f"{offset:>10.2f}{np.mean(s['iou']):>11.4f}{np.median(s['iou']):>12.4f}"
              f"{np.mean(s['ratio']):>12.4f}{s['lost']:>7}")

    usable = {o: np.mean(s["iou"]) for o, s in stats.items() if s["iou"] and s["lost"] == 0}
    if usable:
        best = max(usable, key=usable.get)
        print(f"\nbest offset without destroying any instance: {best:.2f} px "
              f"-> IoU {usable[best]:.4f} (baseline {usable.get(0.0, float('nan')):.4f})")


if __name__ == "__main__":
    main()
