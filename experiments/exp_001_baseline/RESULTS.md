# Experiment 001 — results

Run: Kaggle kernel `cosmicwitness/sol-exp001-baseline`, Tesla T4, 2026-08-31.
Early stopped at epoch 46 after 12 epochs without gain; best epoch 34.

## Headline

| Metric | Value |
|---|---|
| Validation PQ (best epoch, subset signal) | 0.3005 |
| Validation PQ (full 180 records) | **0.2957** |
| SQ — segmentation quality | 0.6293 |
| RQ — recognition quality | 0.4700 |
| TP / FP / FN | 673 / 866 / 652 |

Per-image PQ spread: p10 0.139, p25 0.220, p50 0.301, p75 0.384, p90 0.477.
Only 4 of 180 records score exactly 0, so the model is doing something
everywhere; it is the quality of the instance decomposition that is poor.

## The hypothesis held, and the failure is the interesting part

The predicted result landed in the 0.15–0.30 band the experiment predicted. What
matters is *which* term is short.

**Mask quality is respectable. Recognition is not.** SQ 0.63 says that when a
prediction is matched to a filament, it covers it reasonably. RQ 0.47 says
matching succeeds less than half the time. PQ is their product, so recognition
is what is costing the score.

**This is not over-prediction.** The obvious reading of FP 866 > TP 673 is that
the model hallucinates filaments. The per-image counts say otherwise:

    predictions per image  8.6
    ground truth per image 7.4

The model emits very nearly the right *number* of objects. It is drawing them in
the wrong places — or, far more likely given the design, drawing one filament as
two.

**Why one filament becomes two.** A filament broken into several dark patches by
seeing conditions is one object in the ground truth. Connected-component
labelling has no way to know that and returns one instance per patch. Each
fragment then holds roughly half of the true filament, so its IoU against that
filament falls below the 0.5 matching threshold and *both* fragments count as
false positives while the filament itself counts as a false negative. One
physical error produces three penalty terms. That is the arithmetic that turns
respectable pixels into RQ 0.47.

`src/ceiling_analysis.py` already showed that connected components are not
inherently the problem — fed *perfect* masks it reaches RQ 0.998. The
fragmentation is being introduced by the model's pixel predictions, and no
post-processing rule downstream of them can undo it.

## What this settles

Improving this pipeline means improving pixel-level segmentation until filaments
stop breaking apart — fighting the symptom. A model that predicts instances
directly never creates the error in the first place, because instance identity is
an output rather than something inferred from connectivity afterwards.

exp_002 is that model. This experiment's value is that it makes the case
quantitatively instead of by assertion.

## Leaderboard

Submitted as reference 55911499. Score recorded in `docs/leaderboard-analysis.md`
once returned.
