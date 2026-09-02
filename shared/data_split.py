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


def make_temporal_split(
    annotation_path: str, val_fraction: float = 0.15, buffer_days: int = 7,
    seed: int = SPLIT_SEED
) -> Split:
    """Split by contiguous time blocks, with a buffer between the folds.

    Why the grouped split is not enough
    -----------------------------------
    `make_split` guarantees that no photograph appears in both folds, which
    removes the annotator-duplicate leak. It does not remove the *temporal* leak,
    and that one is larger. A filament survives on the disk for days to weeks,
    and GONG images the Sun continuously from six stations, so two observations a
    day apart usually contain the same physical filaments wearing slightly
    different seeing.

    Measured on the grouped split: 43% of validation observations have a training
    observation within one day, 64% within two, 94% within a week. Validation is
    therefore scoring the model partly on filaments it trained on, and reads
    optimistically — which is consistent with roughly a third of validation gains
    reaching the leaderboard.

    A per-observation buffer cannot fix this: only 6 of 106 validation
    observations sit more than a week from any training observation, so
    enforcing a gap by exclusion would leave nothing to validate on. Contiguous
    blocks are the way — hold out whole stretches of the archive, and drop the
    observations inside `buffer_days` of a boundary rather than assigning them.

    The archive spans 2011 to 2022, so blocks are cut on date order and the
    buffer discards only the seam.
    """
    import datetime as _dt

    stem_to_ids, _ = load_records(annotation_path)

    def observed(stem: str) -> _dt.date:
        return _dt.datetime.strptime(stem[:8], "%Y%m%d").date()

    ordered = sorted(stem_to_ids, key=observed)
    n_val = max(1, round(len(ordered) * val_fraction))

    # A single contiguous block would tie the fold to one part of the solar
    # cycle, and activity varies enormously across it. Several blocks spread
    # through the archive keep both folds representative.
    n_blocks = 5
    block = n_val // n_blocks
    rng = np.random.default_rng(seed)
    starts = sorted(rng.choice(len(ordered) - block, size=n_blocks, replace=False))

    val_idx: set[int] = set()
    for start in starts:
        val_idx.update(range(start, min(start + block, len(ordered))))

    val_stems = [ordered[i] for i in sorted(val_idx)]
    val_dates = [observed(s) for s in val_stems]

    train_stems = []
    for index, stem in enumerate(ordered):
        if index in val_idx:
            continue
        date = observed(stem)
        # Discard the seam rather than train on it.
        if any(abs((date - v).days) <= buffer_days for v in val_dates):
            continue
        train_stems.append(stem)

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
