"""Build a memory-mapped 1024px cache of images, filament masks and spine maps.

Why spines
----------
exp_001 measured the failure that caps this task: recognition quality 0.470
against a design ceiling of 0.998, caused by one filament being emitted as
several connected components. A fragment holds roughly half its filament, so its
IoU falls under the 0.5 matching threshold and the single physical error is
charged three times — both fragments as false positives, the filament as a false
negative.

Every one of the 8199 annotations carries a `spine`: the polyline running along
the filament's central axis. A filament whose mask is broken into three patches
by seeing conditions still has exactly one spine. Predicting the spine alongside
the mask therefore supplies what connected components cannot infer — how many
filaments are present and which pixels belong together — and instance identity
comes from seeding on spines rather than from mask connectivity.

Why a cache
-----------
Eight TPU cores consume batches faster than eight dataloader workers can decode
2048px JPEGs and rasterise polygons. Decoding once into flat uint8 memmaps moves
that cost out of the training loop; afterwards a worker only slices an array.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from shared.data_split import assert_disjoint, make_split

NATIVE = 2048


def render_targets(annotations: list[dict], size: int, spine_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise the filament mask and the spine map for one record."""
    scale = size / NATIVE
    mask = np.zeros((size, size), dtype=np.uint8)
    spine = np.zeros((size, size), dtype=np.uint8)

    for annotation in annotations:
        for ring in annotation.get("segmentation") or []:
            if len(ring) < 6:
                continue
            points = (np.asarray(ring, dtype=np.float32).reshape(-1, 2) * scale).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)

        polyline = annotation.get("spine")
        if polyline and len(polyline) >= 4:
            points = (np.asarray(polyline, dtype=np.float32).reshape(-1, 2) * scale).astype(np.int32)
            # A one-pixel curve is far too sparse a target to learn against, and
            # too fragile to threshold at inference. Drawing it with thickness
            # gives the loss something to grip while keeping distinct filaments
            # separated, which is the property instance seeding depends on.
            cv2.polylines(spine, [points], isClosed=False, color=1, thickness=spine_width)

    return mask, spine


def build(annotations_path: str, images_dir: str, out_dir: str, size: int, spine_width: int) -> None:
    with open(annotations_path, encoding="utf-8") as fh:
        coco = json.load(fh)

    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    meta = {r["id"]: r for r in coco["images"]}

    split = make_split(annotations_path)
    assert_disjoint(split)
    print(split.summary(), flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for fold, ids in (("train", split.train_image_ids), ("val", split.val_image_ids)):
        ids = [i for i in ids if i in meta]
        count = len(ids)
        images = np.lib.format.open_memmap(out / f"{fold}_image.npy", mode="w+",
                                           dtype=np.uint8, shape=(count, size, size))
        masks = np.lib.format.open_memmap(out / f"{fold}_mask.npy", mode="w+",
                                          dtype=np.uint8, shape=(count, size, size))
        spines = np.lib.format.open_memmap(out / f"{fold}_spine.npy", mode="w+",
                                           dtype=np.uint8, shape=(count, size, size))

        # Several records share one photograph; decoding it once per fold pass is
        # wasteful, so the most recent decode is kept.
        cached_stem, cached_image = None, None
        for position, image_id in enumerate(ids):
            stem = meta[image_id]["file_name"]
            if stem != cached_stem:
                raw = cv2.imread(str(Path(images_dir) / stem), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise SystemExit(f"cannot read {Path(images_dir) / stem}")
                cached_image = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
                cached_stem = stem
            images[position] = cached_image
            masks[position], spines[position] = render_targets(by_image[image_id], size, spine_width)

            if (position + 1) % 100 == 0 or position + 1 == count:
                print(f"  {fold} {position + 1}/{count}", flush=True)

        for array in (images, masks, spines):
            array.flush()
        (out / f"{fold}_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"{fold}: {count} records | mask {masks.mean() * 100:.2f}% "
              f"| spine {spines.mean() * 100:.3f}% of pixels", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--spine-width", type=int, default=5)
    args = parser.parse_args()
    build(args.annotations, args.images, args.output, args.size, args.spine_width)


if __name__ == "__main__":
    main()
