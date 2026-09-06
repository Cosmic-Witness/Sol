"""Validate a submission, send it, and wait for the score.

Submission slots are the scarcest resource in a competition, and this one has a
hard rejection rule: no two predicted masks in an image may share a pixel. Every
check that can be run locally is run before a slot is spent.

    python scripts/submit.py outputs/submission.csv -m "exp_002: yolo11m-seg 1280"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from shared.utils import check_no_overlap

COMPETITION = "filament-segmentation-2026"
N_TEST_IMAGES = 180


def validate(csv_path: Path) -> None:
    """Everything that can be checked without the leaderboard."""
    frame = pd.read_csv(csv_path)

    expected = ["filament_id", "segmentation_rle"]
    if list(frame.columns) != expected:
        raise SystemExit(f"columns are {list(frame.columns)}, expected {expected}")

    if frame["filament_id"].duplicated().any():
        duplicates = frame.loc[frame["filament_id"].duplicated(), "filament_id"].head()
        raise SystemExit(f"duplicate filament_id values: {list(duplicates)}")

    if frame["segmentation_rle"].isna().any():
        raise SystemExit("some rows have an empty segmentation_rle")

    # A quoted RLE string is silently accepted by pandas and then misparsed by
    # the scorer, so it is worth catching here rather than in a wasted slot.
    if frame["segmentation_rle"].str.startswith('"').any():
        raise SystemExit("some RLE strings are quoted; write them raw")

    stems = frame["filament_id"].str.rsplit("_", n=1).str[0]
    covered = stems.nunique()
    print(f"rows      : {len(frame)}")
    print(f"images    : {covered} / {N_TEST_IMAGES}")
    print(f"instances : {len(frame) / max(covered, 1):.2f} per image")

    if covered < N_TEST_IMAGES:
        # Not fatal: an image with no filament legitimately contributes no rows.
        # Under Panoptic Quality every missing image is pure false negative, so
        # a large shortfall means the confidence threshold is too high.
        print(f"WARNING: {N_TEST_IMAGES - covered} images contribute no prediction")

    check_no_overlap(str(csv_path))
    print("overlap   : none (masks are pixel-disjoint)")


def submit(csv_path: Path, message: str) -> None:
    subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMPETITION,
         "-f", str(csv_path), "-m", message],
        check=True,
    )


def latest_score(timeout_s: int = 900) -> None:
    """Poll until the newest submission stops reporting as pending."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", COMPETITION, "-v"],
            capture_output=True, text=True,
        )
        rows = [r for r in result.stdout.splitlines() if r.strip()]
        if len(rows) > 1:
            newest = rows[1]
            if "pending" not in newest.lower():
                print(newest)
                return
        time.sleep(20)
    print("still pending after the timeout; check with `kaggle competitions submissions`")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("-m", "--message", required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate only")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"no such file: {args.csv}")

    validate(args.csv)
    if args.dry_run:
        print("dry run: not submitted")
        return

    submit(args.csv, args.message)
    latest_score()


if __name__ == "__main__":
    main()
