"""Convert the competition's COCO annotations into a YOLO segmentation dataset.

Why a separate representation
-----------------------------
exp_001 predicts a binary filament/background map and recovers instances with
connected components. That design has a measured ceiling: feeding perfect ground
truth through it reaches PQ 0.76 at 512 and 0.87 at 1024, because splitting on
connectivity cannot rejoin a filament that the sky broke into pieces. A model
that predicts instances directly has no such ceiling, which is why exp_002 moves
to YOLO segmentation.

Two properties of this dataset drive the conversion and are easy to get wrong.

Fragmented filaments
    A single physical filament is often several disconnected dark patches, and
    the ground truth calls that one object. COCO can store that as several
    polygons under one annotation, while the YOLO label format allows exactly
    one polygon per instance; writing one line per ring would teach the model
    that every fragment is its own filament, which is the over-fragmentation
    failure that destroys recognition quality.

    Measured on the actual file: all 8199 annotations carry exactly one ring,
    and none is RLE-encoded, so `merge_multi_segment` never fires on this
    dataset. It is kept as a guard, not as a working part of the pipeline — if a
    future annotation release does ship multi-ring instances, the converter
    stitches them instead of silently splitting them.

Repeated observations
    296 of the 707 photographs were annotated independently by two or three
    people, and each batch is its own COCO `image_id`. They are kept as separate
    training samples, as the competition notes advise, by naming each sample
    after its `image_id` rather than its `file_name`. The canonical split still
    groups them by photograph, so no photograph appears in both folds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from shared.data_split import assert_disjoint, make_split

# Ultralytics rasterises labels with cv2.fillPoly; the competition scorer uses
# pycocotools. On this dataset the two disagree at IoU 0.90, with the Ultralytics
# raster 11% larger in area — filaments are nearly all perimeter, so a one-pixel
# convention difference is a large fraction of the object. Training against the
# fatter convention teaches the model to over-trace every filament.
#
# Shrinking the polygon by half a pixel before it is written reconciles them:
# IoU 0.8979 -> 0.9592 and area ratio 1.1115 -> 0.9864, measured over 250
# instances by experiments/exp_006_diagnostics/src/polygon_offset.py. The
# correction has to happen in coordinate space, because on a raster the finest
# available operation is a full pixel and that overshoots.
DEFAULT_POLYGON_OFFSET = 0.5


def merge_multi_segment(segments: list[list[float]]) -> np.ndarray:
    """Stitch several polygons into one closed path.

    Each polygon is connected to the next through the pair of vertices that lie
    closest together, so the seam runs along the shortest possible bridge and
    adds the least spurious area. This mirrors the approach Ultralytics uses for
    COCO conversion; it is reimplemented here so the conversion does not depend
    on an internal API that may move between releases.
    """
    rings = [np.array(s, dtype=np.float64).reshape(-1, 2) for s in segments]
    if len(rings) == 1:
        return rings[0]

    merged = rings[0]
    remaining = rings[1:]
    while remaining:
        # Find the ring, and the vertex pair, closest to the path built so far.
        best = (None, 0, 0, np.inf)
        for index, ring in enumerate(remaining):
            distances = np.linalg.norm(merged[:, None, :] - ring[None, :, :], axis=-1)
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[i, j] < best[3]:
                best = (index, int(i), int(j), float(distances[i, j]))

        index, i, j, _ = best
        ring = remaining.pop(index)
        # Enter the ring at its nearest vertex, walk it fully, and come back, so
        # the result stays a single closed path.
        rotated = np.concatenate([ring[j:], ring[:j + 1]], axis=0)
        merged = np.concatenate([merged[:i + 1], rotated, merged[i:]], axis=0)
    return merged


def shrink(polygon: np.ndarray, offset: float) -> list[np.ndarray]:
    """Offset a polygon inward by `offset` pixels, in coordinate space.

    Shapely's buffer moves every edge by the same perpendicular distance, which
    is what a rasterisation boundary offset is. Scaling toward the centroid would
    not do: a filament is long and curved, so that would move its ends far more
    than its middle.

    A buffer can split a thin shape at a narrow waist, or erase it entirely.
    Both are failure modes worse than the fatness being corrected:

    - **Splitting** turns one filament into two instances. Under Panoptic Quality
      that is charged three times — both fragments as false positives and the
      filament as a false negative — and it teaches the model the
      over-fragmentation this pipeline is otherwise built to avoid. Measured on
      this dataset, an unguarded 0.5px buffer splits 42 of 8199 instances. Only
      the largest surviving piece is kept, so the instance count is preserved.
    - **Erasure** loses the filament outright, so the original polygon is
      returned instead: tracing half a pixel too wide is better than not tracing.
    """
    if offset <= 0 or len(polygon) < 3:
        return [polygon]
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return [polygon]

    shape = Polygon(polygon)
    if not shape.is_valid:
        shape = shape.buffer(0)          # repairs self-intersecting traces
    if shape.is_empty:
        return [polygon]

    shrunk = shape.buffer(-offset)
    if shrunk.is_empty:
        return [polygon]

    geoms = [shrunk] if shrunk.geom_type == "Polygon" else list(shrunk.geoms)
    # One filament in, one filament out. Keeping every fragment would inflate the
    # instance count and train the model to split filaments at their waists.
    geoms = [g for g in geoms if g.exterior is not None and len(g.exterior.coords) >= 3]
    if not geoms:
        return [polygon]
    largest = max(geoms, key=lambda g: g.area)
    return [np.asarray(largest.exterior.coords, dtype=np.float64)]


def write_label(path: Path, polygons: list[np.ndarray], width: int, height: int) -> int:
    """Write one YOLO segmentation label file. Returns the instance count."""
    lines = []
    for polygon in polygons:
        normalised = polygon.copy()
        normalised[:, 0] /= width
        normalised[:, 1] /= height
        # Clip rather than drop: a vertex a pixel outside the frame is a
        # rounding artefact of the annotation, not a reason to lose the filament.
        np.clip(normalised, 0.0, 1.0, out=normalised)
        if len(normalised) < 3:
            continue
        coords = " ".join(f"{v:.6f}" for v in normalised.reshape(-1))
        lines.append(f"0 {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def build(annotations: str, images_dir: str, output_dir: str, val_fraction: float,
          polygon_offset: float = DEFAULT_POLYGON_OFFSET) -> None:
    with open(annotations, encoding="utf-8") as fh:
        coco = json.load(fh)

    by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    meta = {record["id"]: record for record in coco["images"]}

    split = make_split(annotations, val_fraction=val_fraction)
    assert_disjoint(split)
    print(split.summary(), flush=True)

    fold_of = {i: "train" for i in split.train_image_ids}
    fold_of.update({i: "val" for i in split.val_image_ids})

    root = Path(output_dir)
    for fold in ("train", "val"):
        (root / "images" / fold).mkdir(parents=True, exist_ok=True)
        (root / "labels" / fold).mkdir(parents=True, exist_ok=True)

    source = Path(images_dir)
    counts = {"train": 0, "val": 0}
    instances = {"train": 0, "val": 0}
    empty = 0

    for image_id, record in meta.items():
        fold = fold_of.get(image_id)
        if fold is None:
            continue

        width = record.get("width") or 2048
        height = record.get("height") or 2048

        polygons = []
        for annotation in by_image.get(image_id, []):
            segmentation = annotation.get("segmentation")
            # An RLE-encoded annotation has no polygon to convert; the dataset
            # is polygon-based, so this is a guard rather than an expected path.
            if not segmentation or isinstance(segmentation, dict):
                continue
            rings = [s for s in segmentation if len(s) >= 6]
            if not rings:
                continue
            merged = merge_multi_segment(rings)
            polygons.extend(shrink(merged, polygon_offset))

        if not polygons:
            empty += 1
            continue

        # The image is shared by every annotation batch of this photograph, so it
        # is symlinked under the record's own name rather than copied 1154 times.
        destination = root / "images" / fold / f"{image_id}.jpeg"
        if not destination.exists():
            original = source / record["file_name"]
            if not original.exists():
                raise SystemExit(f"missing image referenced by annotations: {original}")
            destination.symlink_to(original.resolve())

        written = write_label(root / "labels" / fold / f"{image_id}.txt", polygons, width, height)
        counts[fold] += 1
        instances[fold] += written

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: filament\n",
        encoding="utf-8",
    )

    print(f"train records: {counts['train']} ({instances['train']} instances)", flush=True)
    print(f"val records  : {counts['val']} ({instances['val']} instances)", flush=True)
    if empty:
        print(f"skipped {empty} records with no usable polygon", flush=True)
    print(f"wrote {data_yaml}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--polygon-offset", type=float, default=DEFAULT_POLYGON_OFFSET,
                        help="inward offset in px reconciling Ultralytics' cv2 "
                             "rasterisation with the scorer's pycocotools one; "
                             "0 disables")
    args = parser.parse_args()
    build(args.annotations, args.images, args.output, args.val_fraction,
          args.polygon_offset)


if __name__ == "__main__":
    main()
