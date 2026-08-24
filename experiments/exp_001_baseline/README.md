# Experiment 001 — Baseline: semantic segmentation, then connected components

## Hypothesis

A conventional encoder-decoder trained on binary filament masks, followed by
connected-component labelling, produces a working end-to-end pipeline and a
Panoptic Quality in the 0.15–0.30 band. The purpose is a trustworthy baseline
and a measured failure mode, not a competitive score.

## Method

| Stage | Choice | Reason |
|---|---|---|
| Conditioning | disk detection, limb-darkening flattening, CLAHE | filament contrast varies with radius; flattening makes it uniform |
| Resolution | 512 x 512 for training, 2048 for scoring | a T4 holds 512 comfortably; scoring must match the leaderboard frame |
| Architecture | U-Net++ with an ImageNet `efficientnet-b4` encoder | standard, well-behaved, no novel failure modes to debug |
| Loss | 0.5 BCE (pos_weight 8) + 0.5 soft Dice | filament pixels are about 1% of the disk |
| Optimiser | AdamW, lr 1e-4, cosine decay, 2-epoch warmup | warmup protects the pretrained encoder from the random decoder |
| Instances | threshold, close, connected components, min area 150 | simplest rule that yields disjoint masks |

## Why the split is made on the photograph

The annotation file holds 1154 records but only 707 distinct observations. 296
observations were annotated independently by two or three people, and each batch
carries its own `image_id`. Splitting on `image_id` would validate the model on
photographs it trained on. `shared/data_split.py` splits on `file_name` and
stratifies by GONG site.

Resulting fold sizes: **601 observations / 974 records** train, **106
observations / 180 records** validation.

## Measured design ceiling

Before training, the ground-truth mask was fed through this exact pipeline
(downsample to `image_size`, threshold, close, label components) to find the
best PQ the design could reach with a perfect segmentation model. Over 60
validation records, `src/ceiling_analysis.py` reports:

| Resolution | Closing | PQ | SQ | RQ | FP | FN |
|---|---|---|---|---|---|---|
| 512 | 1 | **0.7642** | 0.7660 | 0.9977 | 0 | 2 |
| 512 | 5 | 0.7589 | 0.7607 | 0.9977 | 0 | 2 |
| 512 | 9 | 0.7481 | 0.7508 | 0.9965 | 0 | 3 |
| 1024 | 1 | **0.8692** | 0.8702 | 0.9988 | 0 | 1 |
| 1024 | 5 | 0.8430 | 0.8449 | 0.9977 | 0 | 2 |

Three findings, each of which changed a decision:

1. **Over-fragmentation is not the problem.** RQ reaches 0.998 with plain
   connected components, and the predicted-to-truth instance ratio is 1.00.
   Filaments in MAGFiLO are already separated as connected components. The
   instance-grouping work that experiment 002 was expected to need is not where
   the score is lost.
2. **Everything is lost in SQ, meaning boundary precision.** The whole gap from
   1.0 comes from IoU on matched pairs.
3. **Closing hurts.** It thickens masks and costs more IoU than the fragments it
   rejoins return. `closing_kernel` is therefore set to 1, which disables it.

Resolution is the strongest lever available: 512 to 1024 moves the ceiling from
0.76 to 0.87. Experiment 001 still trains at 512, because the target band of
0.35 to 0.45 sits far below the 0.76 cap, so the model's own error will dominate
long before the resolution cap binds. Resolution is recorded as the first thing
experiment 002 should spend GPU time on.

## How to run

Training happens on Google Colab. See `notebooks/colab_exp001.ipynb`. Locally:

```bash
# 1. build the conditioning cache (once, ~15 min for 707 observations)
python -m experiments.exp_001_baseline.src.prepare_cache --config experiments/exp_001_baseline/config.yaml

# 2. train, resuming automatically after a disconnect
python -m experiments.exp_001_baseline.src.train --config experiments/exp_001_baseline/config.yaml

# 3. score the best checkpoint over the whole validation fold
python -m experiments.exp_001_baseline.src.evaluate \
    --config experiments/exp_001_baseline/config.yaml \
    --checkpoint experiments/exp_001_baseline/checkpoints/best.pt

# 4. write and validate a submission
python -m experiments.exp_001_baseline.src.predict \
    --config experiments/exp_001_baseline/config.yaml \
    --checkpoint experiments/exp_001_baseline/checkpoints/best.pt \
    --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --output experiments/exp_001_baseline/outputs/submission.csv --tta
```

## Results

Not yet run. `RESULTS.md` is written after the first complete training run.
