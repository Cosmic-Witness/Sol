"""Shared utilities reused by every experiment in this repository.

Modules
-------
utils          Seeding, RLE encoding, Panoptic Quality, overlap handling.
preprocessing  H-alpha specific image conditioning (disk mask, limb correction, CLAHE).
data_split     The single canonical train/validation split used by all experiments.

Nothing here may import from `experiments/`. The dependency direction is one way,
so that a change to one experiment can never alter the metric another was scored with.
"""
