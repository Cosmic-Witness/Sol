# Leaderboard analysis — 2026-08-31

Reconnaissance performed before committing GPU time, so that effort is aimed at
the real frontier rather than at a number that cannot be reached honestly.

## The public leaderboard has a hole in it

465 teams. The score distribution is not shaped like a competition:

| Score | Teams |
|---|---|
| 0.55 | 12 |
| 0.40 | 5 |
| 0.39 | 6 |
| 0.38 | 24 |
| 0.37 | 30 |
| 0.36 | 39 |
| 0.35 | 31 |
| 0.34 | 45 |
| 0.33 | 32 |

A dense, continuous mass from 0.30 to 0.40 — the shape independent work
produces — then **nothing whatsoever between 0.40 and 0.55**, then twelve teams
sharing 0.55 to the last displayed digit. An empty interval that wide is not
what a difficulty cliff looks like; a cliff still has stragglers on its face.
Twelve teams landing on an identical score means twelve teams running one
artefact.

## The evaluator was demonstrably broken

The most-upvoted notebook in the competition is not a solution. It is
`artkomissar/please-fix-the-leaderboard`, a responsible disclosure showing that
a submission containing **five images out of 180, each with a completely empty
mask and zero predicted foreground pixels, scored a public 1.00**. The 175
omitted images drew no penalty.

Two things follow, and they point in opposite directions:

- That specific exploit is dated 2026-07-28 and targeted the **old Dice-based
  metric**. The competition has since moved to Panoptic Quality, which penalises
  false negatives directly and so cannot reward withholding predictions. The
  current top score is 0.55 rather than 1.00, which is consistent with the hole
  having been closed.
- But the episode establishes that this leaderboard's numbers have been
  detached from segmentation quality before. A score here is evidence about the
  scorer as much as about the model.

## Position taken

Two routes to a top score are available and are not being taken.

**The MAGFiLO v1.0 leak.** The public Harvard Dataverse release of MAGFiLO v1.0
overlaps this competition's test set. Training on its annotations would import
test labels directly. `CLAUDE.MD` records this as grounds for disqualification,
and the competition notes it explicitly. Using external H-alpha *imagery* for
self-supervised pretraining is permitted and remains open; using external
*ground truth* for test images is not.

**Evaluator exploitation.** Whatever remains of the scoring weakness, submitting
against it produces a number rather than a segmentation. The final rubric is 70%
quantitative and 30% qualitative, and the qualitative half is judged on a
technical report and on the apparent morphology of predicted masks. A submission
optimised against a scoring artefact has nothing to show either judge.

The working assumption is therefore that **~0.40 is the honest frontier** and
0.55 is not a modelling result. The target is to beat 0.40 by a clear margin
with a real model, and to say plainly where that lands rather than to report a
rank obtained by other means.

## What the 0.55 cluster actually runs

`hdjojo/solar-filament-seg-inference` is public and its author sits in the 0.55
cluster. It is a YOLOv8l-seg model, `nc=1`, inference at `imgsz=2048`,
`conf=0.3`, `iou=0.0`, followed by greedy score-ordered overlap removal.

It carries a real defect. Overlap resolution sorts by

    scores = result.boxes.cls.cpu().numpy()

`cls` is the predicted class index, and the model has exactly one class, so that
array is all zeros and the sort does nothing. Contested pixels go to whichever
mask happens to be last in the raw output rather than to the most confident one.
`boxes.conf` is the field that orders them correctly. exp_002 uses it.

The architectural lesson is worth taking regardless of the defect: the strongest
public approach predicts instances directly rather than segmenting semantically
and splitting on connectivity. That agrees with exp_001's own measured ceiling.


---

# Results log

## exp_001 — U-Net++ semantic + connected components

| | |
|---|---|
| Validation PQ | 0.2957 |
| **Public LB** | **0.26** |
| Rank | ~314 / 466 (13 teams tied at 0.26) |
| Submission | 55911499 |

The validation-to-leaderboard gap is 0.036, small enough that **the validation
fold can be trusted as a proxy**. That matters more than the score itself: it
means thresholds and architectures can be compared offline without spending
submissions, and the grouped split is doing its job — had it leaked, validation
would have read far higher than the leaderboard.

The score itself is what a semantic-then-split design is worth on this task. See
`experiments/exp_001_baseline/RESULTS.md` for why recognition rather than mask
quality is the binding term.

## exp_002 — YOLO instance segmentation, yolo11m-seg @ 1280

| | |
|---|---|
| Epochs | 149 (time-capped at 8.5 h, **not converged**) |
| Mask mAP50 / mAP50-95 | 0.709 / 0.284 |
| **Public LB** | **0.32** |
| Rank | ~227 / 467 (top 49%) |
| Submission | 55917111 |

Moving from semantic-plus-connected-components to direct instance segmentation
bought **+0.06** (0.26 -> 0.32) and moved 91 places. The direction is right and
the margin is smaller than hoped.

Two things are known to be left on the table, and neither is an architecture
problem:

- **The run is undertrained.** Best mask mAP50 came at epoch 149 of 149 and was
  still rising over the final thirty epochs. The clock stopped it mid-climb.
- **The confidence threshold is untuned.** 0.25 was inherited as a guess. Under
  PQ, confidence trades false positives against false negatives directly, and
  the sweep has not run yet.

## Standing assessment

0.32 sits 0.08 below the honest frontier and 0.23 below the leaderboard's top
cluster. Threshold tuning and resolution are each plausibly worth a few
hundredths, so approaching 0.40 is realistic and clearing it is not assured.
Reaching 0.55 by modelling is not in view from here, which is consistent with
this document's opening argument that 0.55 is not a modelling result.

## Threshold tuning is exhausted (negative result)

The sweep over exp_002 ran 7 confidence values x 3 minimum areas on the
validation fold:

| conf | min_area | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.30 | 300 | **0.3736** | 0.6432 | 0.5807 | 721 | 437 | 604 |
| 0.25 | 150 | 0.3714 | 0.6426 | 0.5779 | 751 | 523 | 574 |
| 0.40 | 150 | 0.3657 | 0.6475 | 0.5647 | 626 | 266 | 699 |
| 0.20 | 300 | 0.3629 | 0.6411 | 0.5660 | 780 | 651 | 545 |

Best is conf 0.30 / min_area 300 at PQ 0.3736, against 0.3714 for the inherited
default. **The whole knob is worth +0.0022** — within noise, and not worth a
submission slot. The surface is flat because PQ charges a false positive and a
false negative the same half unit, so trading one for the other moves the
numerator and denominator almost together.

This closes off post-processing as a source of gains and points at the real
constraint.

## The binding constraint is recall, and it is a resolution problem

Comparing the two experiments at their best settings:

| | PQ | SQ | RQ |
|---|---|---|---|
| exp_001 semantic + CC | 0.2957 | 0.629 | 0.470 |
| exp_002 instance seg | 0.3736 | 0.643 | 0.581 |

Instance segmentation delivered exactly what was predicted of it: **+24% on
recognition**, with segmentation quality unchanged. The remaining shortfall is
also recognition, but of a different kind.

At conf 0.20 — the most permissive setting swept, already past the point where
added false positives cost more than the recovered detections are worth — there
are still **545 false negatives out of 1325 ground-truth instances**. Roughly
40% of filaments are not detected at *any* threshold. They are not being scored
away in post-processing; the model never proposes them.

Filament barbs are a few pixels wide at 2048 and sub-pixel at 1280. A structure
that has been resampled below the detector's stride cannot be recovered by
lowering a threshold. That is the case for exp_003 at full resolution, and it is
now an argument from measurement rather than from expectation.

## Spine seeding does not help (negative result)

Every one of the 8199 annotations carries a `spine` polyline, and the plan was
to predict it alongside the mask so instance identity could be seeded from
spines rather than inferred from mask connectivity. The stated motivation was
that one filament broken into several patches still has one spine.

Measured before spending TPU quota, by degrading ground-truth masks and scoring
both decompositions against the true instances
(`experiments/exp_004_spine_tpu/src/spine_ablation.py`):

| erosion | pred/true ratio | CC PQ | spine PQ | delta |
|---|---|---|---|---|
| 0 | 1.00 | 1.0000 | 1.0000 | +0.0000 |
| 3 | 0.99 | 0.8947 | 0.8952 | +0.0005 |
| 5 | 0.80 | 0.5059 | 0.5058 | -0.0001 |
| 7 | 0.66 | 0.2875 | 0.2874 | -0.0001 |

The idea is worth nothing here, and the ablation shows why the premise was
wrong. Under erosion the predicted-to-true ratio falls rather than rises: the
degradation destroys small filaments instead of splitting large ones, so it
never reproduced fragmentation to begin with.

Checking the real numbers rather than the story: exp_001 emitted **8.6**
instances per image against **7.4** in the ground truth, a ratio of 1.16. That
is mild over-segmentation, not the rampant fragmentation the earlier write-up
asserted. The claim that "one filament rendered as two is triply penalised" was
a plausible mechanism promoted to a finding without being tested, and it is
withdrawn.

**The corrected diagnosis.** Matched predictions score SQ 0.63, yet only about
half of predictions and half of the ground truth match at all. Instances are
being emitted in roughly the right number and the right places, and are simply
not accurate enough to clear the IoU 0.5 threshold. That is mask precision, and
resolution and model capacity are what move it — which is what exp_003 was
built to test and what remains untested.
