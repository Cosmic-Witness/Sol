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

## exp_004 — dense segmentation at 1024, trained to convergence on TPU

| | |
|---|---|
| Architecture | U-Net, resnet34 encoder, 1024px, TTA |
| Hardware | TPU v5litepod-8, single process, SPMD across 8 cores |
| Epochs | **300, converged** — val loss flat at 0.1813-0.1815 over the last five |
| Wall clock | 3.8 h (1.4 min/epoch, against the T4's 3.4 min/epoch at 1280) |
| Instances/image | 9.50, against roughly 7.4 in the ground truth |
| **Public LB** | **0.28** |

## The architecture is the ceiling, not the training budget

| run | architecture | training | LB |
|---|---|---|---|
| exp_001 | dense @512 | 46 epochs | 0.26 |
| exp_004 | dense @1024 | 300 epochs, converged | 0.28 |
| exp_002 | YOLO instance @1280 | 149 epochs, still improving | **0.32** |

Doubling the resolution and training the dense line all the way to convergence
bought **+0.02**. An instance model that was cut off mid-climb still beats it by
0.04. Whatever caps the dense approach is not epochs and not resolution, because
both were removed and the score barely moved.

This also corrects the diagnosis recorded above. Mask precision was identified as
the binding constraint, and resolution and capacity as the levers. Resolution was
then doubled with almost no effect. The honest reading is that predicting
instances directly captures something the dense pipeline cannot recover
afterwards, whatever the pixel quality — and that connected components reaching
PQ 1.000 on *ground-truth* masks was never evidence they would do so on predicted
ones.

exp_004 also over-predicts: 9.50 instances per image against about 7.4 in the
ground truth. Under PQ each false positive costs half a unit, which is part of
why converging did not pay.

## Where this leaves the effort

Best remains **exp_002 at 0.32**, rank about 227 of 467. The single untested
idea with evidence behind it is the one that was never allowed to finish:
exp_003, a 2048 fine-tune of exp_002's weights, which keeps the architecture
that demonstrably wins and adds the resolution that the dense line could not
exploit but an instance model may. It needs roughly 6 h of GPU. 4.6 h remain,
and GPU is reserved for other work.

## Center/offset grouping also fails its ablation (negative result)

exp_004's conclusion — that predicting instances directly beats inferring them —
suggested a dense model could still win if its *decoder* predicted instances:
a center heatmap plus per-pixel offsets, the Panoptic-DeepLab formulation, which
is all fixed-shape dense maps and so suits a TPU.

Measured first (`exp_005_center_offset/src/grouping_ablation.py`), with noise
added to the offsets to imitate an imperfect network:

| offset noise (px) | CC PQ | CC RQ | center/offset PQ | center/offset RQ | delta |
|---|---|---|---|---|---|
| 0 | 0.9995 | 1.000 | 1.0000 | 1.000 | +0.0005 |
| 5 | 0.9995 | 1.000 | 0.9938 | 0.997 | -0.0057 |
| 15 | 0.9995 | 1.000 | 0.9694 | 0.989 | -0.0301 |
| 30 | 0.9995 | 1.000 | 0.8336 | 0.932 | -0.1658 |

Grouping degrades steeply once offsets are wrong by a realistic margin, and a
trained network's offsets will be wrong by a realistic margin.

The ablation also shares the defect that made the spine result and the
fragmentation story misleading: **on ground-truth masks connected components
already score 0.9995, so there is no headroom for any decoder to demonstrate a
gain.** Every ablation of this shape is structurally incapable of showing what it
is meant to show. Only predicted masks can settle a decoder question, and
obtaining those costs a training run.

Three decoder ideas have now been tested against ground truth — closing, spine
seeding, center/offset — and all three measured at or below zero. Combined with
exp_004 scoring 0.28 after converging at double resolution, the conclusion is
that **the dense line is finished at roughly 0.28**, and no post-hoc decoder
recovers the gap to 0.32.

## Why work stopped here rather than continuing on TPU

16.9 h of TPU quota remained and was deliberately not spent.

The only remaining idea with positive evidence is exp_003: a 2048 fine-tune of
exp_002's weights, keeping the architecture that measurably wins. It requires
GPU, which is reserved for other work and has 4.66 h left against roughly 6 h
needed.

The TPU-viable alternative would be implementing an instance architecture that
runs under torch_xla — SOLO or CondInst, both dense and fixed-shape. That is a
from-scratch implementation trained on 974 images, competing against a
COCO-pretrained yolo11m that already scores 0.32. It would very probably lose,
and committing 16 h of someone's quota to a hypothesis this repository's own
ablations do not support is the mistake that has already been made twice here.

## TTA is unavailable for YOLO segmentation models (negative result)

Both exp_002 submissions were made without test-time augmentation, which looked
like an oversight worth correcting for free on CPU. It is not correctable that
way. Ultralytics answers `augment=True` on a `-seg` model with

    WARNING: Model does not support 'augment=True', reverting to single-scale prediction

and proceeds. The flag is accepted, silently ignored, and the run produces
output identical to the untriggered case — 1112 rows, matching the tuned
submission that already scored 0.32 exactly. Nothing was gained and there was
nothing new to submit.

Two things worth carrying forward:

- **The failure is silent.** `predict.py` and `tune.py` both expose `--tta`, and
  both have always passed it through to an option the model ignores. Any future
  claim that "TTA is enabled" on this line is false unless the log is checked.
- **exp_004's TTA is real.** That model is a plain PyTorch U-Net and its
  augmentation is implemented directly in `predict.py` as flips of the
  probability field, so it does what it says. The distinction is between a
  library flag and code that was written and can be read.

Averaging predictions over flips for an *instance* model is still possible in
principle, but it requires matching instances across the augmented views before
merging them, which is a real algorithm rather than a flag, and it is unvalidated
here.

## Inference resolution: the cheapest real gain in the project

exp_002 was trained at 1280 and had only ever been run at 1280. Sweeping the
*inference* resolution over the validation fold, with no retraining:

| inference imgsz | val PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| 1280 | 0.3736 | 0.643 | 0.581 | 721 | 437 | 604 |
| 1600 | 0.3965 | — | — | 796 | 513 | 529 |
| **2048** | **0.4064** | 0.670 | 0.607 | 854 | 635 | **471** |

**+0.0328 validation PQ, at zero quota cost.** Larger than anything a night of
GPU and TPU training produced.

The mechanism is the one the recall analysis predicted. False negatives fall
604 -> 471 and true positives rise 721 -> 854: at 1280 a barb a few pixels wide
at native scale is sub-pixel and the detector never proposes it. False positives
rise too (437 -> 635), but under PQ the recovered detections outweigh them.

Worth noting the model was **trained** at 1280. Running a detector 1.6x above its
training scale normally costs accuracy; here it gains, which says the resolution
deficit was severe enough to dominate the mismatch penalty.

**Public score: 0.33**, rank ~204/483 — a new best, up from 0.32 at rank ~238.

## The validation-to-public gap is widening

| | val PQ | public | gap |
|---|---|---|---|
| exp_002 @1280 | 0.3736 | 0.32 | 0.054 |
| exp_002 @2048 | 0.4064 | 0.33 | 0.076 |

Validation gained 0.033 and the leaderboard gained 0.010. The tuning was done on
the validation fold, so some of that is selection pressure on 106 photographs,
and the two sets are not identically distributed. Treat validation PQ on this
project as an optimistic and increasingly loose upper bound rather than a
predictor: a third of a validation gain reaching the leaderboard is the observed
exchange rate, not a shortfall to be explained away.

## What this implies for exp_003

exp_003 retrains at 2048 rather than merely inferring there, which removes the
train/inference mismatch instead of paying it. The evidence for it is now
measured rather than argued: resolution is worth real PQ on this model, and the
gain arrives specifically as recovered recall.

## The human ceiling: inter-annotator PQ is 0.3371

296 observations in this file carry two or three independent annotations (411
once, 145 twice, 151 three times). Scoring every annotator against every other
on the same photograph, 598 pairs, at the same 1024 raster and 150px minimum
area the model is held to:

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| annotator vs annotator | **0.3371** | 0.6341 | 0.5317 | 2122 | 1873 | 1865 |
| exp_002 @2048, validation | **0.4064** | 0.6695 | 0.6070 | 854 | 635 | 471 |

The model scores above inter-annotator agreement on all three quantities.

This is not a claim that the model beats experts. A model trained across many
annotators learns their average, and an average sits closer to any individual
than two individuals sit to each other; regression to the mean produces exactly
this. But it bounds what is left to win.

**It reframes the near-miss population.** The obvious reading of 635 false
positives against SQ 0.67 is that masks are poor and a third of those
predictions are real filaments segmented just under the IoU 0.5 threshold.
Human annotators produce 1873 false positives against each other at SQ 0.634 on
the same data. A large share of the near-miss population is two people
disagreeing about where a filament ends and whether a faint patch is one
filament or two — not model error, and not recoverable by better masks.

**It changes what the remaining work should target.** Pushing SQ past 0.67
means fitting the noise between annotators more tightly than annotators fit each
other. The principled response is not a better mask loss but better targets:
fuse the 296 multiply-annotated observations into consensus masks and train on
those. That raises the ceiling instead of chasing it.

It also partly explains the widening validation-to-public gap. Validation is
scored against one annotator's opinion per record; so is the leaderboard. Some of
the difference between 0.4064 and 0.33 is which annotator happened to label the
test set.

**Caveat on the ceiling.** Inter-annotator PQ is not a hard cap on leaderboard
score. The test set is labelled once, and a model trained on consensus can beat
any individual annotator against another individual. The number bounds how much
of the residual is signal, not how high a score can go.

## The metric is confirmed, and our implementation matches it

The competition page is JavaScript-rendered and unreadable to a fetcher, but the
organiser publishes `azimahmadzadeh/self-evaluation-notebook`, which contains the
scoring code itself.

**The leaderboard metric is Panoptic Quality at IoU threshold 0.5:**

    PQ = sum(IoU of TP pairs) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)

Multi-scale IoU is **not** the leaderboard metric. It appears among the final
rubric's diagnostics alongside Dice distributions and one-to-many relations, but
the leaderboard number is PQ alone. Barb completeness therefore matters exactly
as much as its effect on IoU, and no more.

`tests/test_pq_matches_official.py` pins our implementation against theirs. The
two differ in form: the organiser marks **every** GT/prediction pair above the
threshold as a true positive, while `shared.utils.compute_pq` performs a greedy
one-to-one match. They are equivalent whenever masks are pixel-disjoint, by a
counting argument — two disjoint predictions cannot each cover more than half of
one ground-truth mask — and both sides are disjoint in practice: the scorer
rejects overlapping submissions, and ground truth is disjoint by construction
(0 of 107 sampled images contain overlapping GT instances).

Verified across 25 parametrised cases: counts identical, PQ equal to float32
precision. The one construction where they diverge — two identical predictions
against one ground truth, which the organiser scores as two true positives and
we score as one plus a false positive — cannot reach the leaderboard, and is
pinned by its own test so the assumption stays visible.

**Consequence:** every threshold, resolution and post-processing decision taken
on validation has been measuring the leaderboard's quantity, not an approximation
of it.

## Consensus fusion does not produce a better target (negative result)

The human-ceiling measurement suggested the response was better labels rather
than better masks: fuse the 296 multiply-annotated observations into consensus
instances and train on the agreement. `shared/consensus.py` implements it —
instances linked across annotators at a permissive IoU 0.25 (0.5 would split
genuine agreements, since annotators only reach SQ 0.63 against each other),
then reduced by pixel-wise majority vote, then painted disjoint.

Scored naively it looks decisive:

| | PQ | SQ | RQ |
|---|---|---|---|
| annotator vs annotator | 0.3361 | 0.6348 | 0.5296 |
| annotator vs consensus | 0.6558 | 0.8061 | 0.8134 |

That comparison is circular. The consensus is built from those same annotators,
so of course they agree with it. The honest test is leave-one-out, available on
the 151 three-annotator observations: build the consensus from two annotators and
score it against the third, who contributed nothing to it.

| | PQ | SQ | RQ |
|---|---|---|---|
| held-out vs a single other annotator | 0.3287 | 0.6329 | 0.5193 |
| held-out vs consensus of the other two | 0.3291 | 0.6350 | 0.5183 |
| **delta** | **+0.0004** | +0.0021 | -0.0010 |

**Two annotators averaged predict a third no better than one annotator does.**

This is a stronger statement than "consensus training will not help", though it
is that too. It says the disagreement between annotators is not random error
around a true segmentation, which averaging would reduce. It is structured
ambiguity about what a filament *is* — where a faint extension stops being part
of the spine, whether two nearby patches are one object or two. Averaging two
opinions does not move you closer to a third, because there is no single answer
they are all noisy measurements of.

## What that implies about the score

| | PQ |
|---|---|
| inter-annotator agreement | 0.3371 |
| **this model's public leaderboard score** | **0.33** |
| this model's validation, against one annotator | 0.4064 |

The model agrees with whichever annotator labelled the test set about as well as
two human experts agree with each other. That is not proof of a ceiling — the
leaderboard's top cluster sits at 0.55, and a model can exceed pairwise human
agreement by learning the central tendency of many annotators, which is what the
validation figure of 0.4064 reflects. But it does mean the remaining headroom
below 0.55 is smaller than the raw gap suggests, and that gains from fitting
mask boundaries more tightly are competing with an ambiguity the labels
themselves do not resolve.

## The validation split leaks through time, and it is large

The canonical split groups by photograph, so no observation appears in both
folds and the annotator-duplicate leak is closed. An earlier note in this
document claimed the temporal leak was "bounded at 10%". That measured the wrong
thing — observations sharing a *calendar day* — and is withdrawn.

A filament survives on the disk for days to weeks, and GONG images the Sun
continuously from six stations. Two observations a day apart usually contain the
same physical filaments under different seeing. Measured against temporal
proximity rather than date equality:

| buffer | validation observations with a training observation inside it |
|---|---|
| ±0 days | 10.4% |
| ±1 day | **43.4%** |
| ±2 days | 64.2% |
| ±3 days | 79.2% |
| ±7 days | 94.3% |

**43% of validation observations have a training observation within one day.**
Validation has been scoring the model partly on filaments it trained on, which
is a direct explanation for why roughly a third of validation gains reach the
leaderboard.

A per-observation buffer cannot fix it: only 6 of 106 validation observations sit
more than a week from any training observation, so enforcing a gap by exclusion
would leave nothing to validate against. `make_temporal_split` instead holds out
five contiguous blocks spread across the 2011-2022 archive — several rather than
one, because a single block would tie the fold to one part of the solar cycle,
over which activity varies enormously — and discards observations within the
buffer of a block boundary rather than assigning them.

| split | train | val | val within 1d of train | within 7d |
|---|---|---|---|---|
| grouped (current) | 601 obs / 974 rec | 106 obs / 180 rec | 43.4% | 94.3% |
| temporal, 7-day buffer | 578 obs / 940 rec | 105 obs / 173 rec | **0.0%** | **0.0%** |

The leak closes completely for 23 training observations and one validation
observation. That is a very cheap fix for a defect that has been inflating every
validation number in this document.

**What this does and does not invalidate.** Comparisons made *within* the
grouped split — erosion against no erosion, 2048 against 1280 — are still valid
rankings, because both arms saw the same leak. The absolute values are
optimistic, and the leaderboard has been the honest check throughout. The next
training run should use the temporal split so its numbers mean what they say.

## exp_005 — mask erosion: 0.33 to 0.36, rank 206 to 85

The largest gain of the project, from post-processing alone. No retraining, no
GPU, no new detections.

| | validation PQ | public | rank |
|---|---|---|---|
| exp_002 @1280 | 0.3736 | 0.32 | ~240/486 |
| @2048 inference | 0.4064 | 0.33 | ~206/486 |
| **@2048 + 1px erosion, conf 0.35** | **0.4404** | **0.36** | **~85/486 (top 17.5%)** |

### The hypothesis was backwards

The experiment was built to test whether masks should be **grown**. Ultralytics
thresholds mask prototypes at 0.5, and for a structure a few pixels wide the
prototype field is smooth enough that the tails of a thin ridge fall below the
threshold first — so thin barbs should be systematically eroded, and dilation
should recover them.

The measurement says the opposite, monotonically. Across conf 0.30-0.50 and
growth -4 to +3, every dilation loses and every erosion up to -1 wins:

| grow | PQ (conf 0.35) | SQ | TP | FP |
|---|---|---|---|---|
| -4 | 0.1156 | 0.5953 | 200 | 535 |
| -2 | 0.3702 | 0.6507 | 708 | 456 |
| **-1** | **0.4404** | **0.6843** | 845 | 456 |
| 0 | 0.4169 | 0.6708 | 829 | 514 |
| +3 | 0.2016 | 0.6024 | 448 | 905 |

The optimum is interior in both dimensions — confidence peaks at 0.35 with 0.30
and 0.40 both lower, growth peaks at -1 with -2 and 0 both lower — so it is a
real optimum and not a grid boundary.

### Why erosion wins

Two effects compound, and the counts separate them.

**Masks are systematically too large.** SQ rises 0.6708 to 0.6843 under erosion:
the masks that already matched now overlap their ground truth *better*. YOLO
traces filaments slightly fat, so trimming a pixel tightens every match.

**Marginal detections are culled rather than charged.** FP falls 514 to 456 while
TP falls only 829 to 845 — it rises, in fact. Eroding shrinks borderline blobs
below the 300px minimum area, so they are dropped before scoring instead of
costing half a unit of denominator each.

The near-miss framing that motivated this was right about *where* the score
leaks and wrong about the direction of the fix. Over-large masks were both
diluting real matches and manufacturing false positives.

### The exchange rate is not a constant

| change | validation | public | transfer |
|---|---|---|---|
| resolution 1280 to 2048 | +0.033 | +0.010 | 30% |
| 1px erosion | +0.034 | +0.030 | 88% |

Two changes of near-identical validation size transferred at 30% and 88%. The
"roughly a third reaches the leaderboard" rule recorded earlier does not hold and
should not be used to forecast. A plausible reading is that resolution gains
concentrate on faint, ambiguous structures — precisely where annotators disagree
and the leak between temporally adjacent observations helps most — while erosion
is a systematic geometric correction that applies identically to every image and
therefore survives the distribution shift intact.

## The training targets are rasterised fatter than the scorer measures

`experiments/exp_006_diagnostics/src/rasterisation.py`.

Turning a polygon into a binary mask requires a convention about which boundary
pixels are inside, and the three common libraries disagree. On a compact object
that is a rounding error. A filament is nearly all perimeter — measured
perimeter-to-area ratio 0.216 — so the same disagreement is a large fraction of
its area.

Over 400 sampled instances at 2048:

| pair | mean IoU | median |
|---|---|---|
| pycocotools vs cv2 | 0.8744 | 0.8794 |
| pycocotools vs PIL | 0.9146 | 0.9181 |
| cv2 vs PIL | 0.8970 | 0.9050 |

| rasteriser | mean area | vs pycocotools |
|---|---|---|
| pycocotools | 2250.6 | — |
| PIL | 2379.0 | +5.71% |
| cv2 | 2429.6 | **+7.96%** |

**The two sides of this project use different conventions.** Ultralytics
rasterises training targets with `cv2.fillPoly` (`ultralytics.data.utils.polygon2mask`).
The organiser's scorer uses `pycocotools`. Measured end to end on the actual
pipeline:

    IoU(ultralytics target, pycocotools scorer) = 0.9016
    area ratio                                  = 1.1077

**Training targets are 10.8% larger in area than what the leaderboard measures.**
The model is faithfully reproducing masks that are fat by construction, and no
amount of finer loss supervision would fix that, because the offset is in the
targets rather than in the optimisation.

### But it does not fully explain the erosion gain

The obvious next step is to check whether the 1px erosion that lifted the
leaderboard is simply this offset being undone. It is not — it overshoots:

| | IoU vs scorer | area ratio |
|---|---|---|
| target as rasterised | 0.9016 | 1.1077 |
| target eroded 1px | 0.8945 | 0.9004 |
| target eroded 2px | 0.7010 | — |

One pixel of erosion moves the area from 11% too large to 10% too small, and IoU
against the scorer's rasterisation slightly *falls*. Correcting the convention
alone would take roughly half a pixel.

So the +0.03 leaderboard gain from eroding predictions is not pure convention
correction. Two candidate contributions remain, and they are separable:

- the convention offset, worth about half a pixel of the trim, and
- the minimum-area filter, which discards blobs that erosion shrinks below 300px
  before they can be scored as false positives.

The second is a confound in the erosion result as originally reported: FP fell
514 to 456, and some unknown share of that is the area filter rather than the
boundary. Isolating it requires re-running the sweep with the area filter off.

### What this changes

The planned paid retrain was justified by the hypothesis that coarse loss
supervision (`mask_ratio=4`) causes the fat masks. That hypothesis now has a
competitor with direct evidence behind it, and the competitor predicts the
retrain would not help — the targets carry the offset regardless of the
supervision resolution.

The cheaper fix is to rasterise the training targets to match the scorer's
convention, which is a change to target generation rather than to the loss, and
costs nothing to try on free quota.

## The rasterisation offset is shape-dependent, and post-hoc correction is quantised

Two follow-ups sharpen the finding above.

**The offset is not a constant.** Over 498 instances, the target/scorer area
ratio correlates with perimeter-to-area at **r = +0.926**:

| instance size | area ratio | perimeter/area |
|---|---|---|
| Q1, 132-670 px | 1.155 | 0.290 |
| Q2, 670-1237 px | 1.121 | 0.230 |
| Q3, 1237-2512 px | 1.101 | 0.195 |
| Q4, 2512-27957 px | 1.074 | 0.142 |

Thin filaments are 15% too fat, chunky ones 7%. That is what a boundary effect
must do — the disagreement lives on the perimeter, so its cost as a *fraction of
area* scales with perimeter per unit area. The earlier reasoning that "a uniform
bias implies a constant cause" had the right suspect but the wrong premise: the
bias is systematic, not uniform.

This partly rescues the global 1px erosion. A fixed-pixel erosion also removes
area in proportion to perimeter, so it is already scaled the right way; it is the
magnitude that overshoots, not the shape dependence.

**Post-hoc correction cannot go below one pixel.** Attempting a sub-pixel trim by
thresholding the distance transform does nothing: on a binary mask the transform
is quantised, and `dt > 0.25`, `> 0.5` and `> 0.75` all return the original mask
because boundary pixels sit at distance exactly 1.0.

| trim | IoU vs scorer | area ratio |
|---|---|---|
| none | 0.8926 | 1.1187 |
| 0.25 / 0.5 / 0.75 px | 0.8926 | 1.1187 |
| 1.0 px | 0.8872 | 0.8948 |

So one pixel is the finest available correction after the fact, and it moves the
area from 12% too large to 11% too small. The convention offset wants about half
a pixel and there is no way to spend half a pixel on a raster.

**Which points the fix at target generation rather than at inference.** Shrinking
the polygon *coordinates* before Ultralytics rasterises them has no quantisation
floor — an inward offset of a fraction of a pixel is well defined in coordinate
space. That is a change to `prepare_yolo.py`, costs nothing, and is the correct
form of the fix.

## Status of the `mask_ratio` hypothesis

Checked directly in Ultralytics 8.4.137, `v8SegmentationLoss`:

```python
if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
    # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
    proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
```

The commented-out line is the old behaviour — targets downsampled to the
prototype grid, which would have made `mask_ratio` inert for the loss. It is dead
code. The live line upsamples the **prototypes** to the target resolution
instead, so `mask_ratio=1` genuinely supervises at full resolution in this
version.

The knob works. Whether it is the right knob is now doubtful for a different
reason: the rasterisation offset lives in the targets, and finer supervision
against a fat target reproduces the fat target more faithfully.

## Resolution is maxed out at native (negative result)

Inference resolution was the first real gain of the project, 1280 to 2048 taking
validation PQ 0.3736 to 0.4064. The obvious question is whether it keeps going.
It does not — upsampling past the native frame degrades:

| imgsz | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| **2048** | **0.4064** | 0.6695 | 0.6070 | 854 | 635 | 471 |
| 2560 | 0.3993 | 0.6727 | 0.5936 | 772 | 504 | 553 |
| 3072 | 0.3624 | 0.6728 | 0.5387 | 728 | 650 | 597 |

The mechanism is visible in the split. Segmentation quality is flat across all
three, 0.6695 to 0.6728 — the masks that *are* found are no better at 3072 than
at 2048, which is what upsampling should do, since it adds no information the
sensor did not record. Recognition falls sharply, 0.607 to 0.539, and false
negatives rise 471 to 597.

So beyond native the model simply finds fewer filaments: objects are inflated
past the scales the detector learned, and the anchor-free head has a limited
range of object sizes it responds to. The earlier gain from 1280 to 2048 was
recovering structures that downsampling had destroyed; there is nothing left to
recover once every recorded pixel is present.

**2048 is the operating point.** Resolution is finished as a source of gains
without retraining at a different scale.

## Calibrated emission loses to raw confidence (negative result)

The emission rule says a candidate is worth predicting when
P(match) * E[IoU] > 0.5 * PQ, about P > 0.32 at the current operating point, and
that detector confidence is not P(match) because it scores the box rather than
whether the mask will clear IoU 0.5. A gradient-boosted model was fitted on mask
area, elongation, distance from the limb and solidity to estimate P(match)
directly, evaluated out-of-fold by GroupKFold over photographs so no candidate
was scored by a model that had seen its photograph.

3686 candidates, 37.7% of which match at IoU > 0.5.

| gate | threshold | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| probability | 0.32 | 0.4040 | 0.6783 | 0.5956 | 866 | 717 | 459 |
| probability | 0.45 | 0.4266 | 0.6812 | 0.6263 | 796 | 421 | 529 |
| **confidence** | **0.35** | **0.4432** | 0.6828 | 0.6491 | 850 | 444 | 475 |

**Raw confidence wins by 0.017 PQ.** The theory about where to threshold was
sound — the probability gate does peak near where the arithmetic says it should —
but the estimator is worse than the baseline it was built to beat. The
geometric features add nothing beyond what confidence already encodes, and the
out-of-fold predictions are noisier than the raw score.

Worth noting what the honest evaluation cost: scored on candidates rather than
grouped by photograph, this would have looked like a win. The consensus
experiment taught that lesson at +0.32 apparent versus +0.0004 real; the same
control applied here turns an apparent improvement into a measured regression.

Detector confidence stays the gate.

## The fix: a half-pixel inward polygon offset at target generation

The rasterisation mismatch is correctable, and in coordinate space rather than on
the raster. Measured over 250 instances
(`experiments/exp_006_diagnostics/src/polygon_offset.py`), shrinking each polygon
before Ultralytics rasterises it:

| inward offset (px) | mean IoU vs scorer | area ratio |
|---|---|---|
| 0.00 (current) | 0.8979 | 1.1115 |
| 0.25 | 0.9455 | 1.0388 |
| 0.35 | 0.9546 | 1.0185 |
| **0.50** | **0.9592** | **0.9864** |
| 0.65 | 0.9451 | 0.9542 |
| 1.00 | 0.8757 | 0.8758 |

**Half a pixel takes agreement from IoU 0.898 to 0.959 and the area ratio from
1.111 to 0.986.** That is the offset the 1px erosion overshoot implied, arrived at
independently.

The correction has to be in coordinate space. On a raster the finest available
operation is one whole pixel — the distance transform is quantised, so no
sub-pixel erosion exists — and one pixel overshoots. A polygon buffer has no such
floor.

Shapely's buffer is used rather than scaling toward the centroid, because a
buffer moves every edge by the same perpendicular distance, which is what a
rasterisation boundary offset is; centroid scaling would move a long curved
filament's ends far more than its middle.

### Two failure modes the naive version has

An unguarded buffer **split 42 of 8199 instances** at their narrow waists. Under
PQ that is charged three times — both fragments as false positives, the filament
as a false negative — and it would train the model toward exactly the
over-fragmentation this pipeline is built to avoid. Only the largest surviving
piece is kept, and the instance count comes back to 6874 train and 1325
validation, matching the original file exactly.

A buffer can also erase a thin filament entirely. There the original polygon is
returned: tracing half a pixel too wide is better than not tracing at all.

`prepare_yolo.py` now applies the offset by default, so the next training run
learns targets in the convention the leaderboard actually measures. This is free,
needs no library patch, and is the change that supersedes the paid `mask_ratio`
experiment.

## The erosion gain is 95% boundary correction

The erosion sweep always ran with a 300px minimum-area filter, so two mechanisms
were tangled in its result: a boundary correction, and the filter quietly
discarding blobs that erosion shrank below the threshold. Re-running with the
filter disabled separates them.

| | grow 0 | grow -1 | gain |
|---|---|---|---|
| with 300px filter | 0.4169 | 0.4404 | **+0.0235** |
| without filter | 0.4153 | 0.4375 | **+0.0223** |

**The boundary effect accounts for 95% of the gain.** The area filter supplies
about 5%.

The decisive number is SQ, which is the mean IoU over *matched* pairs and
therefore cannot be moved by discarding unmatched predictions. With the filter
off it rises **0.6707 to 0.6837**. Masks that already matched now fit their
ground truth better, which is a boundary correction and nothing else.

With the filter off, erosion improves every count at once:

| | TP | FP | FN |
|---|---|---|---|
| grow 0 | 833 | 533 | 492 |
| grow -1 | **861** | **505** | **464** |

More true positives, fewer false positives, fewer false negatives. That is what
a genuine boundary improvement looks like: near-misses cross the IoU 0.5
threshold, converting an FP and an FN into a TP simultaneously.

## The chain, closed

1. Ultralytics rasterises training targets with `cv2.fillPoly`; the scorer uses
   `pycocotools`. The targets are **11% larger in area**, IoU 0.898.
2. The model learns to reproduce those targets, so its predictions are fat in the
   same convention.
3. Eroding predictions by one pixel corrects the boundary, and that correction is
   **95% of the +0.03 leaderboard gain** — not an artefact of the area filter.
4. One pixel is the finest post-hoc correction available and it overshoots; the
   offset wants half a pixel.
5. A **0.5px inward polygon offset** at target generation reconciles the
   conventions properly — IoU 0.898 to 0.959, area ratio 1.111 to 0.986 — with no
   quantisation floor and no library patch.

Step 5 supersedes step 3: the trained fix is per-pixel and exact, where the
post-hoc trim is global and blunt. It is free, and it belongs in the next
training run rather than in a paid experiment.

The paid `mask_ratio=1` experiment is withdrawn. Finer loss supervision against a
target that is 11% too fat reproduces the fat target more faithfully; the defect
was never in the supervision resolution.

## Where the mask error actually lives

The claim under test: that SQ 0.68 means masks are wrong deep inside the object,
making it a representational failure that no boundary work can reach. Measured
over 124 matched pairs at the shipped operating point, bucketing every
disagreement pixel by its distance to the ground-truth boundary:

| distance from GT boundary | FP px | FN px | share | cumulative |
|---|---|---|---|---|
| 0-1 | 14668 | 13580 | 41.4% | 41.4% |
| 1-2 | 7687 | 7797 | 22.7% | **64.2%** |
| 2-3 | 4154 | 4879 | 13.3% | 77.4% |
| 3-5 | 2554 | 3319 | 8.6% | 86.0% |
| 5-8 | 1361 | 1724 | 4.5% | 90.5% |
| 8+ | 5062 | 1384 | 9.5% | 100% |

**64% of the disagreement is within two pixels of the boundary, 77% within
three.** It is overwhelmingly a rim effect. Only 22.6% lies deeper than 3px, so
the representational-failure reading is a minority of the error.

Converting that to score, with disagreement at 0.472x the intersection:

| eliminate | SQ | PQ at RQ 0.639 |
|---|---|---|
| rim only, <=2px | 0.8554 (+0.176) | 0.4347 -> **0.5474** |
| everything <=3px | 0.9036 (+0.224) | -> 0.5783 |
| interior only, >3px | 0.7324 (+0.053) | -> 0.4687 |

**The rim is worth three times the interior.** Perfect boundaries would be worth
+0.11 PQ; perfect interiors +0.03.

### But this does not vindicate the approach taken so far

The rim holds most of the error *and* post-hoc correction can barely touch it.
A global 1px erosion applies the same trim to every mask regardless of whether
that mask was one pixel fat, three pixels fat, or already correct, and one pixel
is the quantisation floor. It captured about 0.023 of an available 0.11.

So both readings were partly wrong. The claim that boundary work is nearly
exhausted is false — the boundary is where 64% of the error is. The claim that
this justified more post-hoc geometry is also false — the blunt instrument is
spent, and what remains needs a model that predicts boundaries per-pixel.

That is the argument for a second-stage refiner, and it is a stronger argument
than the one originally offered for it. Not "the error is deep inside where
refinement is the only reach", but "the error is at the rim, the rim is worth
+0.11 PQ, and a global constant recovers a fifth of it".

### One signal that does support the low-rank reading

The deepest band is lopsided: 5062 FP pixels against 1384 FN beyond 8px, a
3.7:1 skew towards over-prediction, where every other band is near parity. Far
from any true boundary the model adds mask rather than missing it, which is what
leakage between instances sharing a smooth 32-prototype basis would look like.
It is 9.5% of the error, not the main event, but it is the part a refinement
stage on isolated crops would remove by construction.

## exp_008/009 — the refiner loses to a one-pixel erosion (negative result)

A second-stage U-Net was trained to redraw each instance's boundary on a 256px
native-resolution crop, given the image and the coarse mask. The motivation was
sound: 64% of the model's error is within two pixels of the boundary and worth
+0.11 PQ, and a global erosion constant recovers only 0.023 of it.

It trained well. Validation IoU on its own task went from 0.7208 to **0.8529**,
converged, 138 minutes across 8 TPU cores. Scored against real detector output:

| configuration | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| detector, no erosion | 0.4169 | 0.6708 | 0.6214 | 829 | 514 | 496 |
| detector + refiner @0.4 | 0.4296 | 0.6727 | 0.6385 | 847 | 481 | 478 |
| detector + refiner @0.6 | 0.4322 | 0.6733 | 0.6419 | 847 | 467 | 478 |
| **detector + 1px erosion** | **0.4404** | **0.6843** | 0.6436 | 845 | 456 | 480 |

**It loses by 0.0082 to a morphological operation with no parameters.**

It is not a no-op — mean IoU between refined and coarse masks is 0.7925, so it
is moving boundaries substantially — and it does beat raw detector output by
+0.015, nearly matching erosion on recognition quality. It fails specifically on
segmentation quality, 0.6733 against 0.6843.

### Why: it was trained on a model of the error, not the error

The training pairs were synthetic. Ground truth was dilated, translated and
smoothed to imitate the detector's measured failure modes, on the reasoning that
harvesting real pairs would cost hours of inference and buy nothing the
annotations did not already contain.

That reasoning was wrong, and this result is what it cost. What harvesting buys
is exactly the thing synthesis cannot provide: the *real* joint distribution of
detector error. The refiner learned to undo dilation-plus-blur because that is
what it was shown, and the detector's actual mistakes are correlated with image
content — faint filaments near the limb, neighbouring instances competing for
prototypes — in ways a hand-specified corruption does not reproduce.

The simulation was calibrated on *severity* (coarse IoU 0.72 against the
detector's real SQ 0.679) and that was mistaken for calibration on *character*.
Matching the average amount of damage says nothing about matching its structure.

### What this does not overturn

The rim analysis stands: 64% of error is within two pixels of the boundary and
eliminating it is worth +0.11 PQ. The refiner is the right shape of solution and
the target is real. What failed is the training data, which is fixable.

The honest next step is the one originally proposed and originally dismissed:
run the detector over the training photographs, match each output to its ground
truth, and train on those pairs. That is a few hours of CPU in a kernel, and it
is now clearly justified rather than clearly wasteful.

## Stacking the refiner with erosion is much worse (and the first instinct was right)

Stacking was initially refused on the argument that both corrections address the
same fat-mask bias. That was then overturned by noting the refiner reaches SQ
0.6733 against erosion's 0.6843 — apparent under-correction, so a further trim
might be owed. Measured across four thresholds:

| configuration | PQ | SQ | RQ |
|---|---|---|---|
| detector + 1px erosion (shipped) | **0.4404** | 0.6843 | 0.6436 |
| refiner @0.6 alone | 0.4322 | 0.6733 | 0.6419 |
| refiner @0.5 + erosion | 0.3787 | 0.6453 | 0.5868 |
| refiner @0.6 + erosion | 0.3711 | 0.6431 | 0.5770 |
| refiner @0.8 + erosion | 0.3427 | 0.6340 | 0.5405 |

Eroding a refined mask costs 0.06 PQ. The original instinct was correct and the
revision was wrong: the refiner does **not** under-correct. It produces masks of
about the right size whose shape is simply less accurate than the eroded
originals — lower SQ at comparable extent, not lower SQ from being too fat.
Reading a single scalar difference as evidence about *direction* was the error.

The threshold optimum is also genuinely interior now, 0.4305 / 0.4322 / 0.4309 /
0.4267 across 0.5 to 0.8, so the earlier boundary win at 0.6 was not a grid
artefact.

The refiner loses to a parameterless morphological operation across all eight
configurations tested. It stays out of the pipeline until it is retrained on real
detector-versus-truth pairs rather than synthetic damage.

## Dihedral TTA also loses (negative result)

Eight rotations and reflections of each frame, masks inverse-transformed back,
clustered across views and voted per pixel. The reasoning was that the earlier
resolution negative was about *scale* — inflating objects past the sizes the
anchor-free head learned — while rotation and reflection leave scale untouched
and the solar disk has no canonical orientation.

| | PQ |
|---|---|
| single view, shipped config | **0.4404** |
| dihedral TTA, best of 12 fusion settings | 0.4035 |

8715 view-masks over 106 photographs, about 10 per view. The augmentation is
valid; the fusion is what loses. Two mechanisms are plausible and were not
separated:

- **Clustering errors.** Instances are not aligned across views — one view splits
  a filament another keeps whole — so a greedy IoU link either merges distinct
  filaments or fragments one, and both are expensive under PQ.
- **Voting erodes.** Requiring half the contributing views to agree on a pixel
  trims the boundary, on top of a model whose masks already needed trimming, and
  the interaction with the existing erosion was swept but never disentangled.

The honest reading is that this tests *my fusion implementation*, not test-time
augmentation as an idea. A cleaner design would fuse soft mask probabilities
before binarisation rather than voting on already-binarised masks, which is what
Ultralytics' refusal to support `augment=True` on segmentation models forced.
That is worth revisiting; this implementation is not worth keeping.

## The refiner scores 0.35 on the leaderboard, as validation predicted

Submitted as a deliberate regression datum, since validation-to-public transfer
had been erratic enough that a 0.008 deficit did not obviously predict the
leaderboard.

It did. Public 0.35 against 0.36, rank ~136/502 against ~92/502.

| change | validation | public | transfer |
|---|---|---|---|
| resolution 1280 -> 2048 | +0.033 | +0.010 | 30% |
| 1px erosion | +0.034 | +0.030 | 88% |
| refiner (regression) | -0.008 | -0.010 | 122% |

Three data points, transfer between 30% and 122%. The useful reading is not an
average but the sign: **every change has transferred in the direction validation
predicted**, including this one. The magnitude is unreliable — a gain worth 0.033
on validation bought anywhere between 0.010 and 0.030 — but the direction has
never flipped.

That is enough to keep using validation as a *gate* while distrusting it as a
*forecast*. The guard that refuses to submit a losing configuration was right
three times today; overriding it deliberately, once, and labelling the submission
as a regression cost 0.01 and bought the calibration point above.

Best remains **0.36, rank ~92 of 502 (top 18.3%)**.

## exp_013 — what the 456 false positives are actually made of

Three post-processing experiments have now been run against this error. Each
targeted one failure mode, and each assumed the mode it targeted was the
dominant one. None of them checked. `experiments/exp_013_errors/src/decompose.py`
classifies every unmatched prediction and every unmatched truth in the
validation set by its best overlap with the other side, running off the exp_005
candidate cache so it needs neither inference nor GPU. It reproduces the shipped
operating point's PQ to seven decimal places (0.4403668), so the accounting is
against the real thing.

| class | overlap with the other side | false positives | false negatives |
|---|---|---|---|
| near | 0.25-0.50 | 150 (33%) | 153 (32%) |
| graze | 0.10-0.25 | 48 | 28 |
| sliver | 0-0.10 | 34 | 14 |
| **orphan** | **none at all** | **224 (49%)** | **285 (59%)** |

**Half the error is instances one side does not acknowledge exist.** 224
predictions overlap no ground truth anywhere in the image, and 285 ground-truth
filaments are overlapped by no prediction anywhere in the image. No boundary
method can touch either class. The refiner, the one-pixel erosion, the polygon
offset and full-resolution supervision all operate on a mask that is already on
the right object.

Two hypotheses die here. **Splits: 2. Merges: 5.** Out of 936 failures. The
detector is not fragmenting filaments into pieces or fusing neighbours together,
which was the natural reading of a 32-prototype basis shared by long thin
objects. Instance identity is essentially never the problem.

### What each class is worth

Each oracle removes exactly one class and leaves the rest untouched, so the
deltas are the isolated value of solving it.

| oracle | PQ | SQ | RQ | delta |
|---|---|---|---|---|
| measured | 0.4404 | 0.6843 | 0.6436 | — |
| every orphan and sliver FP removed | 0.4883 | 0.6843 | 0.7137 | +0.048 |
| every near miss promoted to a match | 0.4975 | 0.6565 | 0.7578 | +0.057 |
| every false positive removed | 0.5329 | 0.6843 | 0.7788 | +0.093 |
| every false negative removed | 0.5389 | 0.6843 | 0.7875 | +0.099 |
| perfect masks on the matches already made | **0.6436** | 1.0 | 0.6436 | **+0.203** |

The near-miss figure is deliberately pessimistic: it enters each promoted pair
at IoU exactly 0.5, the least it can be worth, which is why SQ falls. At a
realistic 0.65 the same promotion is worth +0.075.

### What this says about where to spend

Mask quality is still the largest single lever at +0.203, and the near-miss
class is reachable by the same work: 150 predictions sitting at mean IoU 0.392
need 0.108 more to become matches. Taking SQ from 0.684 to a modest 0.75 while
carrying two thirds of the near misses over the line puts validation at 0.528 —
which at the observed validation-to-leaderboard offset of 0.078 is a public 0.45.

So boundary work is not exhausted. What is exhausted is *post-hoc* boundary
work, exactly as exp_005 and exp_009 measured. The remaining boundary gain has
to come from the targets and the supervision, which is what exp_010 now carries.

But the orphan class is co-equal at +0.048 for the predictions alone, and it is
not addressable by anything currently planned. Whether it is addressable at all
is the next question, because "overlaps no ground truth" is measured against one
annotator, and two annotators agree at PQ 0.337.

## The plan the decomposition implies

Two defects in the shipped detector are now identified, both in training rather
than in inference, and neither has ever been corrected:

1. **The targets are 11% too fat.** Ultralytics rasterises polygons with
   `cv2.fillPoly`, the scorer uses pycocotools, and over 250 instances the
   training mask agrees with the scored mask at IoU 0.898. A half-pixel inward
   polygon buffer takes that to 0.959.
2. **The detector was trained at 1280 and every submission since has inferred at
   2048.** exp_002's log is unambiguous: `imgsz=1280`, and `149 epochs completed
   in 8.507 hours` — stopped by its `time=8.5` budget, not by convergence or by
   patience. The model's prior over object sizes was fixed at 1280 and is being
   asked about objects 1.6 times larger. That is also the most plausible reading
   of why inference at 2560 and 3072 degrades so sharply.

With the targets corrected, `mask_ratio=1` becomes worth asking for again. The
default computes the mask loss on a 512 grid, where the two-pixel band holding
two thirds of the error is invisible. It was withdrawn on the grounds that finer
supervision against a fat target only reproduces the fat target more faithfully.
That was true, and the polygon offset is what makes it stop being true.

### Allocation

| resource | job | status |
|---|---|---|
| GPU, 30 h from the weekly reset | exp_010: retrain at 2048 on corrected targets with full-resolution mask supervision, resuming across the 12-hour cap | ready |
| CPU, unmetered | exp_015: measure the operating point on validation, predict the test set | dry run against the current checkpoint |
| TPU | exp_012: the refiner retrained on real detector-versus-truth pairs | running |

Training and submission are now separate kernels. The previous arrangement did
both in one, which meant training had to finish inside the 12-hour cap with room
to spare for inference or nothing came out. exp_015 runs on CPU on purpose:
inference over 286 photographs is ninety minutes there and nothing on a T4, and
every GPU hour belongs in the training kernel.

### What would have to be true to reach 0.46

The observed validation-to-leaderboard offset is stable across three
submissions: 0.4064 -> 0.33, 0.4404 -> 0.36. Both gaps are 0.078. A public 0.46
therefore needs a validation PQ near 0.54.

From the oracle table, taking SQ from 0.684 to 0.75 while carrying two thirds of
the 150 near misses over the matching threshold gives 0.528. Adding any part of
the orphan class reaches 0.54. Neither number is out of reach for the two
corrections above, and neither is guaranteed by them.

## exp_014 — the phantom filaments are the label noise floor

The orphan class — instances one side does not acknowledge exist at all — is 49%
of false positives and 59% of false negatives, and nothing in the plan touches
it. But "does not acknowledge" is measured against one annotator, and 47 of the
106 validation photographs carry two or three independent annotations. That
makes the same statistic available for one human against another.

`experiments/exp_014_annotator/src/irreducible.py`, at the shipped operating
point, counting an instance as orphaned when nothing on the other side overlaps
it by even IoU 0.10:

| | orphaned | of | rate |
|---|---|---|---|
| annotator instances, by the detector | 299 | 1325 | **22.6%** |
| annotator instances, by another annotator | 255 | 1315 | **19.4%** |
| detector predictions, by the annotator | 258 | 1301 | 19.8% |

**The detector denies a filament exists at 22.6%, where a second human expert
denies it at 19.4%.** Three points apart. On the reverse direction the detector
is at 19.8% against the human 19.4% — indistinguishable.

Two direct confirmations on the multiply-annotated subset: **87 predictions
orphaned by the reference annotator were drawn by a different annotator** — real
filaments this reference simply did not mark — and **75 missed truths were drawn
by no other annotator either**, labels only one person believed in.

The orphan class is therefore not a detector failure and is very largely
unrecoverable. Of the 299 orphaned truths, roughly 257 are at the rate a human
would also miss, leaving about 42 genuinely attributable to the model. The
+0.048 oracle for deleting orphan false positives, and most of the +0.099 for
deleting false negatives, are oracles over label noise. They should not be
chased.

## exp_016 — the misses are seen and disbelieved, not unseen

The complementary question: the orphan count is measured over predictions that
survived confidence 0.35, but the cached pool runs down to 0.05. Does the
detector propose something at the missed locations and rank it too low?

`experiments/exp_013_errors/src/recall_ceiling.py`, over 1325 validation truths
and a pool averaging 23.4 candidates per photograph:

| | count | share |
|---|---|---|
| some candidate in the pool matches it at IoU 0.5 | 1102 | **83.2%** |
| ...and that candidate is already above confidence 0.35 | 797 | 72.3% of those |
| pool only grazes it, never reaching IoU 0.5 | 195 | 14.7% |
| nothing in the pool touches it at all | 28 | **2.1%** |

**Only 2.1% of the ground truth is invisible to this detector.** The pool's
recall ceiling is 0.832 against a realised recall of 845/1325 = 0.638.

That splits the recoverable error cleanly in two:

- **305 truths have a covering candidate that confidence discards** (median
  confidence of a covering candidate is 0.545, but the tenth percentile is
  0.142). This is a ranking failure.
- **195 truths are proposed but the mask is too wrong to match.** This is mask
  quality, and it is the same lever as the 150 near-miss false positives.

### What a re-ranker would have to achieve

Promoting *k* correct candidates and *m* incorrect ones from the low-confidence
band moves PQ to `(578.2 + 0.65k) / (1313 + 0.5k + 0.5m)`. A promotion that is
right earns a true positive and cancels a false negative; one that is wrong costs
half a false positive.

**Break-even is two wrong per one right — precision above 33%.** The base rate in
the 0.05-0.35 band is about 10.5%, so a re-ranker must be three times better than
chance on whatever subset it selects. exp_005's gradient-boosted model over four
geometric features could not do it. A classifier that sees the image crop might;
it is the same shape of problem as the refiner, on the same infrastructure.

### Where this leaves the allocation

Unchanged, and better justified. Mask quality is the one lever with a clean
target: 195 grazed truths, 150 near-miss false positives, and SQ 0.684 on the 845
matches already made. exp_010 attacks it at the source, and training at 2048
should also raise confidence on the small filaments the 1280 model rates at 0.14,
which is the ranking lever reached from the other side.
