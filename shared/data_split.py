"""The canonical train/validation split, shared by every experiment.

Why this module exists
----------------------
The annotation file holds 1154 image records but only 707 distinct physical
observations. 296 of those observations were annotated independently by two or
three people, and each annotation batch appears as its own `image_id`:

    010101-20160920230134Lh
    010402-20160920230134Lh    <- same photograph, different annotator

Splitting on `image_id` would place two annotations of one photograph on
opposite sides of the split. The model would then be validated on a picture it
trained on, and the validation PQ would read high for the wrong reason.

The split is therefore made on `file_name`, and every annotation batch of an
observation follows its observation into the same fold.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Fixed for the life of the project. Changing it invalidates cross-experiment
# comparisons, so treat it as a constant, not a hyperparameter.
SPLIT_SEED = 2026


@dataclass(frozen=True)
class Split:
    """One train/validation partition, expressed at both levels of identity."""

    train_stems: list[str]
    val_stems: list[str]
    train_image_ids: list[str]
    val_image_ids: list[str]

    def summary(self) -> str:
        return (
            f"train: {len(self.train_stems)} observations / {len(self.train_image_ids)} records | "
            f"val: {len(self.val_stems)} observations / {len(self.val_image_ids)} records"
        )


def instrument_of(file_name: str) -> str:
    """Two-letter GONG site code from a filename.

    `20160920230134Lh.jpeg` -> `Lh` (Learmonth). Sites differ in seeing, optics,
    and exposure, so the code is a useful stratification key.
    """
    return file_name.rsplit(".", 1)[0][-2:]


def load_records(annotation_path: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Read the COCO file and index it for splitting.

    Returns
    -------
    stem_to_ids
        file_name -> list of image_id, one per annotation batch.
    id_to_stem
        image_id -> file_name.
    """
    with open(annotation_path, encoding="utf-8") as fh:
        coco = json.load(fh)

    stem_to_ids: dict[str, list[str]] = defaultdict(list)
    id_to_stem: dict[str, str] = {}
    for record in coco["images"]:
        stem_to_ids[record["file_name"]].append(record["id"])
        id_to_stem[record["id"]] = record["file_name"]
    return dict(stem_to_ids), id_to_stem


def make_split(
    annotation_path: str, val_fraction: float = 0.15, seed: int = SPLIT_SEED
) -> Split:
    """Build the canonical split.

    Observations are stratified by instrument site so that both folds see the
    same mix of image quality. Within each site the order is shuffled with a
    fixed seed, which makes the split identical on every machine and every run.
    """
    stem_to_ids, _ = load_records(annotation_path)

    by_site: dict[str, list[str]] = defaultdict(list)
    for stem in sorted(stem_to_ids):
        by_site[instrument_of(stem)].append(stem)

    rng = np.random.default_rng(seed)
    train_stems: list[str] = []
    val_stems: list[str] = []
    for site in sorted(by_site):
        stems = by_site[site]
        rng.shuffle(stems)
        n_val = max(1, round(len(stems) * val_fraction))
        val_stems.extend(stems[:n_val])
        train_stems.extend(stems[n_val:])

    train_stems.sort()
    val_stems.sort()
    return Split(
        train_stems=train_stems,
        val_stems=val_stems,
        train_image_ids=[i for s in train_stems for i in stem_to_ids[s]],
        val_image_ids=[i for s in val_stems for i in stem_to_ids[s]],
    )


def assert_disjoint(split: Split) -> None:
    """Guard against leakage. Called by the training entry point."""
    overlap = set(split.train_stems) & set(split.val_stems)
    if overlap:
        raise ValueError(f"{len(overlap)} observations appear in both folds: {sorted(overlap)[:5]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print the canonical split summary.")
    parser.add_argument("annotations", help="path to the COCO training annotations")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    split = make_split(args.annotations, args.val_fraction)
    assert_disjoint(split)
    print(split.summary())
