# Experiment 002 — YOLO instance segmentation

## Hypothesis

Predicting filament instances directly scores materially higher than predicting
a binary map and splitting it on connectivity, because the second design has a
ceiling that no amount of model accuracy can lift.

## Why, specifically

`exp_001/src/ceiling_analysis.py` fed **ground truth** through the exp_001
post-processing path — downsample, threshold, close, connected components — and
measured the best PQ that design could reach with a hypothetically perfect
segmentation model:

| Resolution | PQ ceiling |
|---|---|
| 512 | 0.7642 |
| 1024 | 0.8692 |

That is the ceiling for a *perfect* pixel classifier. Every real model sits
below it, and the gap between semantic quality and instance quality is exactly
the part connected components cannot fix: one filament broken into several dark
patches by seeing conditions is one object in the ground truth and several
objects to a connectivity rule.

The strongest public solution on this leaderboard is a YOLO segmentation model.
That is independent corroboration, not the reason for the choice.

## Method

| Stage | Choice | Reason |
|---|---|---|
| Labels | COCO polygons → YOLO segmentation | direct instance supervision |
| Split | canonical grouped split, by photograph | 296 photographs carry 2–3 independent annotations; splitting on `image_id` would validate on trained-upon pictures |
| Samples | one per annotation record (1154) | competition notes advise treating independent annotations of one photograph as separate samples |
| Model | `yolo11m-seg`, `nc=1` | one class; medium is the largest that trains at 1280 within a session |
| Resolution | 1280 | barbs are a few pixels wide at 2048, so resolution buys recall, but epoch count matters more than the last pixels of detail on a T4 |
| Augmentation | flips, ±15° rotation, brightness only | no hue/saturation on grayscale; the disk has no canonical up direction |
| Mosaic | **off** | pasting four full-disk images into one frame invents limbs and destroys the radial context limb-darkening correction depends on |
| Overlaps | painted by descending `boxes.conf` | see below |

## The confidence-ordering fix

The public 0.55 notebook resolves overlaps with

```python
scores = result.boxes.cls.cpu().numpy()   # all zeros when nc=1
```

so its greedy painting is ordered arbitrarily. Panoptic Quality is decided by
IoU against matched ground truth, so giving contested pixels to an arbitrary
mask instead of the confident one costs IoU on both instances involved.
`predict.py` sorts by `boxes.conf`.

Masks are also thresholded *after* being resized to 2048, not before, so the
boundary is quantised once rather than twice. PQ's segmentation-quality term is
a boundary measurement, so this is not cosmetic.

## Verification before spending GPU time

`prepare_yolo.py` was run against the real annotation file with placeholder
images:

- 974 train records / 6874 instances, 180 val records / 1325 instances
- 6874 + 1325 = 8199 = every annotation in the file, none dropped
- split reproduces exp_001's documented 601/974 and 106/180 exactly
- 200 sampled label files: no malformed lines, all coordinates within [0, 1],
  polygons of 10–677 vertices

Also measured, and contrary to the assumption the converter was first written
under: **all 8199 annotations carry exactly one polygon ring**, and none is
RLE-encoded. The multi-ring stitching path is therefore a guard against a future
annotation release, not a working part of this pipeline.

## Status

Training. Results and the leaderboard score land in `RESULTS.md`.
