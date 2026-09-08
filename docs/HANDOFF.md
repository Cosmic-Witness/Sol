# Solar Filament Segmentation 2026 — complete handoff

> Revision note: [architecture-review.md](architecture-review.md) corrects several
> architectural and statistical claims below and documents the new exp_032
> chunked-loss implementation. This handoff remains the historical experiment
> record; its proposed explanations and future plan are not verified results.

Everything known about the competition, everything tried, every measured
constant, and a detailed plan for the approach I did not get to. Written for
someone picking this up cold with no access to the prior conversation.

**State at handoff: public 0.36, rank ~112 of 521. Leaders 0.55-0.56. Target was
0.46 and was not reached.**

---

# PART 1 — THE COMPETITION

## Basics

| | |
|---|---|
| Kaggle slug | `filament-segmentation-2026` |
| Full name | Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup) |
| Deadline | 2026-11-15 |
| Account | `cosmicwitness` (Ismail Kasozi) |
| Submissions | 5 per day |
| Leaderboard display | **two decimals** — this matters, see Part 4 |

## The task

Instance segmentation of solar filaments in H-alpha full-disk photographs from
the GONG network. A filament is a dark, elongated, often wiggly feature on the
bright solar disk, typically with thin protrusions ("barbs"). Each must be
segmented as a separate instance with a pixel-precise mask.

Images are 2048x2048 grayscale JPEG. **2048 is the native sensor resolution** —
there is no detail above it, which is why inference above 2048 always degrades.

## Data layout

```
MAGFiLO_1.0_Kaggle_2026/
  train/
    train_images/<YYYYMMDDHHMMSS><SS>.jpeg      # 707 photographs
    MAGFiLO_1.0_Annotations_kaggle2026_train.json   # COCO format
  test/
    test_images/...                              # 180 photographs
```

Filenames encode timestamp and GONG station: `20110120105534Ch.jpeg` is
2011-01-20 10:55:34 at station `Ch`. Six stations: Bh, Ch, Lh, Mh, Th, Uh
(Big Bear, Cerro Tololo, Learmonth, Mauna Loa, Teide, Udaipur).

### The critical annotation structure

**707 unique photographs but 1154 annotation records.** 296 photographs were
annotated independently by two or three different people, and **each annotation
pass appears as its own `image_id`**:

```
010101-20160920230134Lh     <- annotator A
010402-20160920230134Lh     <- annotator B, same photograph
```

Splitting on `image_id` puts two annotations of one photograph on opposite sides
of the split and the model validates on a picture it trained on.
`shared/data_split.py` splits on `file_name` and is the canonical split for the
whole project — **do not change `SPLIT_SEED = 2026`**, it invalidates every
cross-experiment comparison recorded here.

Split sizes: train 601 photographs / 974 records, val 106 / 180.

Train and test are i.i.d.: year histograms agree within a couple of points
(except 2021-22 where test carries about twice the share of 21 photographs), and
all six stations agree within three points. **Distribution shift is ruled out.**

### Do not train on the public MAGFiLO release

The public MAGFiLO v1.0 release on Zenodo contains the test annotations. Also,
its abstract says "1,593 observations" — that counts annotation passes. There
are **958 unique observations**, of which the competition already uses 887. The
apparent extra data is about 71 photographs.

## The metric — Panoptic Quality

```
PQ = SQ x RQ = (sum of IoU over matched pairs) / (|TP| + 0.5|FP| + 0.5|FN|)
```

A prediction matches a truth when IoU > 0.5. Above 0.5 the match is necessarily
one-to-one for disjoint masks (two disjoint predictions cannot each cover more
than half of one truth), so "all qualifying pairs" and "greedy one-to-one"
coincide.

**The organisers pool TP/FP/FN across the whole test set and divide once.** They
do not average per-image PQ. `shared.utils.aggregate_pq` does the same, and
`tests/test_pq_matches_official.py` is a differential test against a faithful
reimplementation of the organisers' `get_pq_score` (25 parametrised cases, all
passing). **This has been verified — do not re-derive it.**

Submissions with overlapping masks are rejected. `shared.utils.paint_panoptic`
resolves overlaps by painting in descending confidence order, each mask keeping
only unclaimed pixels.

## Submission format

```csv
filament_id,segmentation_rle
20110120105534Ch_1,<COCO RLE counts string>
```

`filament_id` is `<image stem>_<N>` numbered from 1. RLE is COCO-format via
`pycocotools`, encoded from a 2048x2048 binary mask.

---

# PART 2 — ACCESS AND INFRASTRUCTURE

## Kaggle

CLI configured; credentials materialised by `.claude/hooks/session-start.sh`
into `~/.kaggle/kaggle.json` on every session start (containers are ephemeral
and this has already proved necessary after several restarts).

**Quota: 30 GPU hours per week, resetting Saturday 00:00 UTC.** TPU quota is
separate (~20h/week) and was barely used. CPU kernels are effectively unmetered
and capped at 12 hours.

**Kernel caps: 12 hours.** A kernel that hits the cap ends in state `CANCEL`,
not `COMPLETE`.

### Kaggle mechanics learned the hard way

1. **A cancelled kernel's output does not mount** when the kernel is listed as a
   `kernel_source` by another kernel. A *completed* kernel's output does. Since
   long training runs end in `CANCEL`, **ship checkpoints as a Dataset**
   (`kaggle datasets create -p <dir> --dir-mode zip`), which always mounts.
2. **Kaggle strips the top directory when unpacking an uploaded archive.**
   `runs/polygon/weights/best.pt` arrives as `polygon/weights/best.pt`. Glob on
   the leaf (`weights/best.pt`), never on a path above it.
3. **Logs are not retrievable while a kernel runs.** `kaggle kernels output`
   returns nothing, `api.kernels_logs()` returns an empty string, and
   `kernels_logs_stream()` yields nothing. Only the web UI shows a live log.
   Design kernels to fail fast and loudly, because you are blind until they end.
4. **The returned log is truncated to the tail.** Setup lines scroll away. Write
   anything you need to diagnose into a JSON artefact instead.
5. **Cancelling a running kernel:** there is no API for it. Push a trivial new
   version of the same kernel; that supersedes and stops the running one.
6. **A kernel can list itself in `kernel_sources`** to read its own previous
   output, but only if that previous version *completed*. Verified by
   `kaggle/selfref/`.

## Other compute

- **TPU v5litepod-8**, free on Kaggle, ~20h/week, currently unused.
  `xmp.spawn` is broken there; a single process sees all 8 devices and SPMD
  sharding works. `experiments/exp_025_verifier/src/train_verifier.py` is a
  working example.
- **RunPod** — account exists, ~$0.61 left. An earlier run wasted $2.39 by
  provisioning with `volumeInGb: 0`, so checkpoints died with the pod.
  **Always attach persistent storage.**
- **An L40S (48 GB) would change everything.** Every training failure in this
  project traced to a single 3.11 GB allocation not fitting in a T4's 16 GB.

---

# PART 3 — WHAT THE MODEL IS

## Current best: `exp_002`

`yolo11m-seg` (Ultralytics 8.4.x), fine-tuned from COCO weights.

| | |
|---|---|
| training resolution | 1280 |
| epochs | 149, stopped by an 8.5-hour `time=` budget |
| inference resolution | **2048** |
| operating point | conf 0.35, `min_area` 300, 1-pixel erosion |
| validation PQ | **0.4404** (SQ 0.6843, RQ 0.6436) |
| public | **0.36** |

Checkpoint: Kaggle kernel output `cosmicwitness/sol-exp002-yolo-seg`,
`checkpoints/best.pt`.

Other checkpoints:
- `cosmicwitness/sol-exp010-session1` (dataset) — 2048-trained, val 0.4274
- `cosmicwitness/sol-exp031-frozen-1536` (dataset) — 1536, annealed, val 0.4343

## Repository layout

```
shared/
  data_split.py     canonical split — do not change the seed
  utils.py          compute_pq, aggregate_pq, paint_panoptic, RLE, submission
  solar_disk.py     Otsu disk detection (measured useless, 0.000% off-disk)
experiments/
  exp_NNN_name/src/ one directory per experiment
kaggle/
  expNNN/           kernel script + kernel-metadata.json per experiment
results/            raw JSON output of every experiment
docs/
  leaderboard-analysis.md   every experiment with its tables
  strategy.md               the reasoning and what would falsify it
  HANDOFF.md                this file
tests/              PQ differential test, prepare_yolo test
CLAUDE.md           the mistakes made here, so they are not repeated
```

Branch: `claude/kaggle-credentials-setup-f7nudy`. PR #1.

---

# PART 4 — MEASURED CONSTANTS

**These cost real compute to establish. Do not re-measure them.**

## The validation-to-leaderboard relationship

| local PQ | public | gap |
|---|---|---|
| 0.4064 | 0.33 | 0.078 |
| 0.4404 | 0.36 | 0.078 |
| 0.4343 | 0.36 | 0.074 |
| 0.4274 | 0.36 | 0.067 |
| 0.3865 | 0.32 | 0.067 |

**Validation predicts the leaderboard within 0.067-0.078**, verified across a
0.04 range including a submission deliberately pushed below the optimum. Judge
every idea on validation before spending a submission.

**A public 0.46 therefore needs a validation PQ near 0.53.**

## The leaderboard quantises

Four genuinely different models — mask agreement only 71-89% between them —
all scored exactly 0.36. Their validation scores span 0.4274 to 0.4411, which is
**0.0137, narrower than one step of a two-decimal display.** Anything smaller
than about 0.06 validation PQ is invisible on the public leaderboard.

## The human ceiling

| | PQ | SQ | RQ |
|---|---|---|---|
| annotator vs annotator | 0.3361 | 0.6348 | 0.5296 |
| **this model** | **0.4404** | **0.6843** | **0.6436** |

The model already agrees with an annotator better than a second annotator does,
on both components.

Treating each annotator as a noisy draw around a latent boundary, a pairwise
disagreement of 0.365 of the union implies about half that per annotator, so a
model predicting the latent truth exactly would score about **SQ 0.82** against
any one annotator — not 1.0. The rim analysis reached 0.855 independently.

**Consensus labels do not help**: on the 151 three-annotator photographs, the
consensus of two predicts the third no better than one annotator does
(+0.0004 PQ).

## Where the error is

Decomposition at the shipped operating point (`results/exp_013_decomposition.json`),
which reproduces PQ to 0.4403668:

| class | overlap with other side | FP | FN |
|---|---|---|---|
| near miss | 0.25-0.50 | 150 | 153 |
| grazing | 0.10-0.25 | 48 | 28 |
| sliver | 0-0.10 | 34 | 14 |
| **orphan** | **none** | **224** | **285** |

**Splits: 2. Merges: 5.** Out of 936 failures. Instance identity is not the
problem — do not build anything to fix fragmentation.

Where the mask disagreement lies: **64% within 2 pixels of the boundary**, 77%
within 3. It is a rim effect, not a deep representational failure.

## Candidate pool coverage

At confidence floor 0.05 (`results/exp_016_recall_ceiling.json`):

- **83.2%** of truths are covered by some candidate at IoU 0.5
- only **72%** of those clear confidence 0.35
- **2.1%** of truths are invisible to the detector entirely

A perfect re-ranker over the existing pool would score **0.6175 local, about
0.55 public** — exactly the top cluster. The recall required for the leading
score is already present in what this detector emits. Selecting it is the
problem, and Part 5 explains why that is much harder than it sounds.

## Rasterisation mismatch

Ultralytics rasterises training polygons with `cv2.fillPoly`; the scorer uses
`pycocotools`. Over 250 instances the training mask is **11% fatter** than the
scored one, IoU 0.898.

| inward polygon offset | IoU vs scorer | area ratio |
|---|---|---|
| 0.00 | 0.8979 | 1.1115 |
| **0.50** | **0.9592** | **0.9864** |
| 1.00 | 0.8757 | 0.8758 |

`prepare_yolo.py` applies the 0.5px offset by default now.

**Critical interaction:** the loss grid resolution in native pixels is
`2048 * mask_ratio / imgsz`. At the Ultralytics default `mask_ratio=4` and
imgsz 2048 that is 2.0 native pixels, and **a half-pixel correction is invisible
on it.** The offset only takes effect with `mask_ratio=1` at 2048, which does not
fit in a T4 (see Part 6).

## Other measured facts

- **Inference resolution peaks at 2048** for every model tried, regardless of
  training resolution. 1792 → 0.4180, 2048 → 0.4274, 2304 → 0.4264, 2560 →
  0.4088, 3072 → 0.3583. The peak is the native frame, not a multiple of the
  training size.
- **CPU inference is bit-for-bit identical to GPU** (same submission MD5). Use
  CPU kernels for inference and keep GPU hours for training. 286 photographs at
  2048 is ~90 minutes on CPU, ~15 on a T4.
- **`min_area` 250-300 and confidence 0.30-0.35 are optimal**, confirmed
  interior on a widened grid.
- **Erosion depends on the targets**: a model trained on uncorrected (fat)
  targets wants a 1-pixel erosion (+0.023 PQ); one trained on corrected targets
  wants none. Always re-sweep this after retraining.
- **0.000% of predictions fall off the solar disk.** Disk masking is worthless.

---

# PART 5 — EVERYTHING TRIED, AND WHY IT FAILED

Fourteen approaches. All negative. Grouped by what they attacked.

## Post-hoc mask correction (five attempts, all failed)

| experiment | method | result |
|---|---|---|
| exp_005 | 1px erosion | +0.023 — **the only one that worked**, and it is in the baseline |
| exp_009 | refiner U-Net on synthetic damage | learned its task (IoU 0.72→0.85), **lost** PQ (0.4322 vs 0.4404), public 0.35 |
| exp_018 | refiner U-Net on real detector errors | learned nothing (best val IoU at **epoch 2**), 0.4098 — worse than not touching the mask |
| exp_017 | sub-pixel cut of the mask logit field | works as a mechanism (+0.018 alone) but no better than the 1px erosion, and does not stack |
| exp_029 | snap boundary to image intensity | **0.9995 IoU on a synthetic filament, destroys real masks** (SQ 0.678→0.654) |

**Why they all fail.** The boundary the metric rewards is not a property of the
image. Real filaments fade into the disk over several pixels; the annotated
boundary is a judgement about where the fade stops counting. exp_029 is the
cleanest proof: snapping the mask onto the *actual physical edge* makes the score
worse, because the physical edge is not where the annotator drew.

**Do not build a sixth mask corrector.**

## Re-ranking and recall (three attempts, all failed)

| experiment | method | result |
|---|---|---|
| exp_005 | gradient boosting on area, elongation, limb distance, solidity | loses to raw confidence by 0.017 |
| exp_019 | gradient boosting on the mask field's fall-off profile | genuinely adds information (AP 0.7248 vs confidence's 0.7150) but promotes nothing useful |
| exp_025-027 | **CNN verifier on 128px image crops** | see below |

**exp_027 is the most important negative result in the project.**

The verifier is a small CNN (827k params, 2 channels: photograph + proposed
mask) trained on 13,413 labelled crops. In the confidence band where promotion
happens it clearly beats confidence:

| below confidence 0.30 | AP |
|---|---|
| base rate | 0.3442 |
| raw confidence | 0.4552 |
| **verifier** | **0.5761** |

Its top 100 picks are **72% correct**; break-even is 33%. And as an emission rule
it fails completely — promoting adds 100 true positives for **496** false ones,
**17% precision**.

**Why: I labelled a crop positive if *any* annotator drew a filament there.** PQ
scores each record against **one** annotator. The verifier learned "would
somebody draw this" (72%) while the metric asks "did *this* person draw it"
(17%). **That 55-point gap is the annotator disagreement, measured directly.**

Relabelling per-record would not fix it. The two classes are the same filaments;
what separates them is which person looked.

**This is why the 0.55 oracle is not reachable by re-ranking.** The covered
truths cannot be selected because they are indistinguishable from candidates
that are equally real and merely unlabelled.

## Ensembling (two attempts, both failed)

| experiment | method | result |
|---|---|---|
| exp_021 | instance-level fusion of two models | +0.0007. Requiring the second model to confirm removes **7 false positives out of 456** — the models make the same errors |
| exp_030 | averaging the two models' masks | +0.0014 SQ, cancelled by RQ. Variance reduction needs independent errors; both models learned the same annotations |

## Training changes (three attempts)

| experiment | configuration | val PQ | public |
|---|---|---|---|
| exp_002 | 1280, uncorrected targets, truncated by a clock | **0.4404** | 0.36 |
| exp_010 | 2048, corrected targets, `mask_ratio=2` fallback, 62/400 epochs, LR never annealed | 0.4274 | 0.36 |
| exp_031 | 1536, corrected targets, `mask_ratio=1`, **50 epochs fully annealed** | 0.4343 | 0.36 |

**Confirmed:** completing the cosine schedule is worth **+0.0069**.
**Not supported:** that training at the inference resolution helps. 1280 remains
the best training resolution measured and 2048 the worst.

**Never actually run: 2048 + corrected targets + `mask_ratio=1` + full anneal.**
It does not fit in a T4. `kaggle/exp031/` is the closest attempt.

## Other dead ends

- **Dense semantic models**: exp_001 U-Net++ @512 scored 0.26; exp_004 U-Net
  resnet34 @1024, converged, with TTA, scored 0.28 — against the instance
  model's 0.32.
- **Dihedral TTA**: 0.4035 against 0.4404. The model is already flip-augmented,
  so flips are near-redundant, and the instance-clustering fusion loses more than
  the averaging gains.
- **Centre/offset grouping (Panoptic-DeepLab style)**: degrades steeply with
  realistic offset noise, and connected components already score 0.9995 on
  ground-truth masks, so the ablation could never show a gain.
- **Spine seeding, closing, disk masking**: all at or below zero.

---

# PART 6 — THE T4 MEMORY WALL

Understanding this saves a day.

Training `yolo11m-seg` at 2048 with `mask_ratio=1` requires a single **3.11 GB**
allocation that a 16 GB T4 cannot provide. Measured facts, each in a clean
process:

| configuration | allocation wanted | free |
|---|---|---|
| 2048, batch 2 | 3.11 GB | 312 MB |
| 2048, batch 1 | 3.11 GB | 314 MB |
| 1792, batch 2 | 2.75 GB | 2.17 GB |
| 1536, batch 2 | 2.02 GB | 1.97 GB |
| 1536, batch 1 | — | **fits** |
| 1280, batch 4 | fits for 5 epochs, then 1.20 GB on a crowded image | — |

**The allocation is independent of batch size and scales with the number of
instances in a batch**, not with the batch dimension. It is the einsum in the
mask loss producing `(n_instances, H, W)` at full resolution. A crowded
photograph blows it up regardless of batch.

Three traps around this:

1. **Freezing the backbone does not help.** It saves optimiser state, not the
   loss allocation. Memory still sat at 13.2 GB.
2. **Test each configuration in a separate process.** `empty_cache()` cannot
   reclaim memory the previous trainer still references, so a loop of attempts
   in one process gives byte-identical failures for configurations that were
   never actually run.
3. **Set `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF` before anything imports
   torch.** Set after `torch.cuda.is_available()`, it silently does nothing.

**On a 48 GB card none of this exists.**

---

# PART 7 — THE MASK2FORMER PLAN

This is the approach I believe in and did not get to run. It is the only
identified path with arithmetic that reaches the target.

## Why this architecture and not another tweak

YOLO's segmentation head predicts each mask as a linear combination of **32
prototype maps at stride 4**, shared across every instance in the image. Two
consequences:

1. **The basis is low-rank and smooth.** Thin, wiggly, barbed structures are
   exactly what a 32-dimensional smooth basis represents worst. Every mask is
   forced into a band-limited subspace.
2. **Instances share the basis**, so far from any true boundary the model adds
   mask rather than missing it — measured as a 3.7:1 excess of false-positive
   over false-negative pixels beyond 8 pixels from a boundary, which is what
   leakage between instances sharing a basis looks like.

Mask2Former predicts a **per-query mask embedding** with no shared low-rank
basis, and decodes masks by dot product against a high-resolution pixel-decoder
feature map. Its masked-attention decoder attends within the predicted region,
which is well suited to elongated structures. It is also trained with a
Hungarian-matched set-prediction loss, so it does not need NMS or confidence
thresholding in the same way — the operating-point sweep that consumes so much
effort here largely disappears.

Supporting evidence: **PatchPerPix and related local-shape methods win FISBe**,
the closest published benchmark (instance segmentation of long-range thin
filamentous structures). The published SOTA on MAGFiLO itself, EdgeAttNet
(arXiv 2509.02964), reports mIoU pairwise 0.6451 — which is *below* this
project's SQ of 0.6843 — but it is semantic segmentation scored on IoU, not PQ,
so it is not a like-for-like comparison and should not discourage anyone.

**Honest expected value.** Mask quality is capped near SQ 0.82 by annotator
disagreement, so the gain has to come from RQ. Current RQ is 0.6436. If
Mask2Former's set prediction reaches RQ 0.75 at the same SQ, PQ is 0.51 local ≈
0.44 public. To reach 0.53 local it needs RQ near 0.78 *and* SQ near 0.70. That
is plausible and not guaranteed. **Nobody should promise 0.46 from this.**

## Implementation, step by step

### Step 0 — the feasibility probe (do this first, ~20 minutes of GPU)

Do not write the training pipeline before knowing what fits. A kernel that:

1. Loads `facebook/mask2former-swin-tiny-coco-instance` and
   `facebook/mask2former-swin-small-coco-instance`.
2. For each of imgsz 800, 1024, 1280, 1536, runs **one forward and one backward
   pass** on a synthetic batch of the right shape with ~15 instances.
3. Prints peak memory for each and which combinations survive.

Run each combination in a **subprocess** (see Part 6, trap 2). Report a table.
This single kernel decides the resolution the rest of the plan uses, and it is
the step whose absence cost this project two days.

Bear in mind filaments are a few pixels wide at 2048. At 800 they may be
sub-pixel and the whole approach fails on resolution grounds — that is what the
probe needs to tell you, so also check what fraction of ground-truth instances
survive downscaling to each resolution with a minimum-area threshold.

### Step 1 — data pipeline

The competition already ships COCO instance annotations, which is exactly
Mask2Former's native format. Use HuggingFace:

```python
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance",
    do_resize=True, size={"height": H, "width": W},
    do_normalize=True, ignore_index=255, do_reduce_labels=False,
)
```

The dataset must yield, per photograph:
- `pixel_values` — the image, normalised
- `mask_labels` — a `(n_instances, H, W)` float tensor of binary masks
- `class_labels` — a `(n_instances,)` long tensor, all zeros (one class)

Build masks from the COCO polygons with `pycocotools.mask.frPyObjects` +
`merge` + `decode` — **the same path `shared/utils.py` uses**, so the
rasterisation convention matches the scorer and the 11%-fat problem from Part 4
never arises. Do not use `cv2.fillPoly`.

Use `shared.data_split.make_split` for the train/val partition. Nothing else.

Images are grayscale; replicate to three channels for the pretrained backbone.

### Step 2 — training

```python
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance",
    num_labels=1, ignore_mismatched_sizes=True,
)
```

Recommended starting point, to be adjusted by what the probe says fits:

| | |
|---|---|
| resolution | the largest the probe allows, ideally >= 1024 |
| batch | as large as fits; gradient accumulation to an effective 8-16 |
| optimiser | AdamW, lr 1e-4 for the decoder, 1e-5 for the backbone |
| schedule | **cosine sized to complete inside the session** — see below |
| epochs | 50-80 |
| augmentation | horizontal and vertical flips, rotation up to 15 degrees, brightness 0.85-1.15. **No mosaic** — it pastes four disks into one frame and invents limbs. **No colour jitter** — the images are grayscale. |

**Size the schedule to the compute you actually have.** The single confirmed
training result in this project is that completing the cosine is worth +0.0069,
and the run that missed it (400 epochs planned, 62 reached, learning rate never
left its peak) underperformed a shorter run that annealed.

Checkpoint every epoch into `/kaggle/working` so the 12-hour cap is a
checkpoint rather than a deadline, and ship the result as a **Dataset**, not as
a kernel output (Part 2, mechanic 1).

### Step 3 — inference and submission

Mask2Former returns per-query masks and scores.
`processor.post_process_instance_segmentation(outputs, threshold=..., target_sizes=[(2048,2048)])`
gives instance masks at native resolution.

Then:
1. Filter by query score.
2. Pass `(score, mask)` pairs through **`shared.utils.paint_panoptic`** to
   guarantee disjointness — the scorer rejects overlaps.
3. Write with `shared.utils.write_submission`, check with
   `shared.utils.check_no_overlap`.

Sweep the score threshold and `min_area` on validation exactly as
`experiments/exp_005_postproc/src/nearmiss.py` does. **Use the same inference
settings in the sweep and the submission** — validating one rule and shipping
another wasted a submission here (CLAUDE.md, entry 5).

Assert the sweep reproduces a known number before trusting it.

### Step 4 — judge it honestly

Do not submit unless validation beats **0.4404**. Validation predicts the
leaderboard within 0.067-0.078, and a submission is a scarce resource. A
validation gain below about 0.06 will not even be visible on a two-decimal
leaderboard.

## What to do if Mask2Former disappoints

In order:

1. **A larger backbone** (swin-base, swin-large) if memory allows. This dataset
   is small — 601 training photographs — so watch for overfitting, but the
   pretrained backbones are strong.
2. **PatchPerPix or a local-shape-descriptor method.** These win FISBe, which is
   the closest analogue to this problem, and they are designed for exactly the
   failure mode YOLO has: thin structures that a low-rank global basis cannot
   represent. Higher build cost.
3. **Raise `nm` above 32 in the YOLO head.** Cheap to try, directly addresses the
   prototype-basis argument, but discards the pretrained mask head, which on 601
   photographs is a real cost.
4. **Boundary-weighted mask loss.** Full-resolution supervision down-weights the
   rim about fourfold relative to the interior (a 1275-pixel filament with a
   200-pixel perimeter has 15.7% of its mask on the rim at 2048, against 62% at
   512). Weighting the loss near the target boundary recovers that. One training
   run, only worth doing once full-resolution supervision actually runs.

---

# PART 8 — THINGS NOT TO DO

Every one of these was tried and measured.

1. **Do not build another post-hoc mask corrector.** Five attempts, five
   failures, and exp_029 shows why: the boundary the metric wants is not in the
   image.
2. **Do not chase the false positives.** They occur at the human disagreement
   rate. The detector proposes filaments the annotator did not mark at 19.8%; a
   second human does it at 19.4%.
3. **Do not build a re-ranker labelled against the union of annotators.** It
   optimises a different question from the metric. 72% versus 17%.
4. **Do not ensemble these two checkpoints.** Their errors are correlated
   because they learned the same labels.
5. **Do not infer above 2048.** The peak is the native frame.
6. **Do not use `optimizer="auto"` in Ultralytics.** It logs
   `ignoring 'lr0=...'` and picks its own. Name the optimiser.
7. **Do not pass a bare `resume=True`.** It routes to `get_latest_run()`, which
   searches Ultralytics' settings directory, not yours. Pass the path.
8. **Do not remove `time=` without replacing the schedule.** Removing it also
   removes Ultralytics' rescaling of the LR schedule to the budget.
9. **Do not let an OOM ladder fall back silently.** exp_010 spent twelve GPU
   hours measuring a configuration whose central change was inert because the
   ladder quietly dropped to `mask_ratio=2`.
10. **Do not cache dense 2048x2048 masks.** ~9 GB per erosion level; keep RLE
    between uses and let `pycocotools.mask.iou` work on RLE directly.
11. **Do not trust a synthetic test that encodes your assumption.** Intensity
    snapping scored 0.9995 synthetic and destroyed real masks.
12. **Do not train on the public MAGFiLO release.** It contains the test
    annotations.

---

# PART 9 — THE HONEST ASSESSMENT

The score did not move: 0.36 at the start of the serious effort and 0.36 at the
end, across five submissions and fourteen approaches.

What the effort did establish, and what has genuine value to whoever continues:

- The metric implementation is verified against the organisers'.
- Validation predicts the leaderboard reliably across a 0.04 range, so ideas can
  be judged for free on CPU.
- The error is decomposed, and the label-noise floor is measured directly rather
  than assumed — including the single cleanest number in the project, the 72%
  versus 17% gap between "is this a filament" and "did this annotator draw it".
- Eleven approaches are eliminated with evidence, so nobody needs to repeat them.
- The T4 memory wall is fully characterised.

What went wrong in the *conduct* of the work, beyond the results: too many
configurations were asserted rather than read back, two diagnoses were announced
before being tested, and a submission was spent on a change my own validation had
already shown to be negligible. `CLAUDE.md` records these in detail; it is worth
reading before starting.

The single most useful thing a successor can do differently: **spend twenty
minutes on a feasibility probe before committing twelve hours of GPU.** Two full
training runs in this project measured the wrong configuration because nobody
checked what actually fit and what was actually set.
