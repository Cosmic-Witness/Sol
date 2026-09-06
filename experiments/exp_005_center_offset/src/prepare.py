"""Cache images and per-pixel instance labels at 1024 for the center/offset model.

What exp_004 established, and what follows from it
--------------------------------------------------
exp_004 trained a dense model to genuine convergence at double exp_001's
resolution and scored 0.28, against 0.32 for an instance model that was cut off
while still improving. Removing both the epoch limit and the resolution limit
bought +0.02. The dense *representation* was therefore not what was failing.

What was failing is the decoder. Connected-component labelling asks "which
pixels touch", and instance identity is not a connectivity property — that
ground-truth masks decompose at PQ 1.000 under CC says nothing about predicted
ones, as exp_004 demonstrated at cost.

A dense model can predict instances directly instead: a center heatmap saying
how many filaments there are and where, and a per-pixel offset vector pointing
at the center each pixel belongs to. Grouping is then learned rather than
inferred, which is the property the YOLO line has and exp_001/exp_004 lacked.
This is the Panoptic-DeepLab formulation, and every tensor in it is a fixed-size
dense map, so it suits both a TPU and XLA's dislike of dynamic shapes.

Caching an instance-label map rather than pre-rendered targets keeps the
representation honest: centers and offsets are derived in the dataset, so the
Gaussian width and offset normalisation can be changed without rebuilding a
multi-gigabyte cache.
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


def render_instances(annotations: list[dict], size: int) -> np.ndarray:
    """One uint16 label map: 0 is background, 1..N are filament instances.

    Instances are drawn largest first so that a small filament overlapping a big
    one keeps its own label rather than being painted over. uint16 caps at 65535
    instances against a maximum of a few dozen here.
    """
    scale = size / NATIVE
    polygons = []
    for annotation in annotations:
        rings = [r for r in (annotation.get("segmentation") or []) if len(r) >= 6]
        if not rings:
            continue
        area = float(annotation.get("area") or 0.0)
        polygons.append((area, [(np.asarray(r, np.float32).reshape(-1, 2) * scale).astype(np.int32)
                                for r in rings]))

    labels = np.zeros((size, size), dtype=np.uint16)
    for index, (_area, rings) in enumerate(sorted(polygons, key=lambda p: -p[0]), start=1):
        cv2.fillPoly(labels, rings, index)
    return labels


def build(annotations_path: str, images_dir: str, out_dir: str, size: int) -> None:
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
        instances = np.lib.format.open_memmap(out / f"{fold}_inst.npy", mode="w+",
                                              dtype=np.uint16, shape=(count, size, size))

        cached_stem, cached_image = None, None
        total_instances = 0
        for position, image_id in enumerate(ids):
            stem = meta[image_id]["file_name"]
            if stem != cached_stem:
                raw = cv2.imread(str(Path(images_dir) / stem), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise SystemExit(f"cannot read {Path(images_dir) / stem}")
                cached_image = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
                cached_stem = stem
            images[position] = cached_image
            labels = render_instances(by_image[image_id], size)
            instances[position] = labels
            total_instances += int(labels.max())

            if (position + 1) % 200 == 0 or position + 1 == count:
                print(f"  {fold} {position + 1}/{count}", flush=True)

        for array in (images, instances):
            array.flush()
        (out / f"{fold}_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"{fold}: {count} records, {total_instances} instances "
              f"({total_instances / max(count,1):.1f} per record)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    build(args.annotations, args.images, args.output, args.size)


if __name__ == "__main__":
    main()
