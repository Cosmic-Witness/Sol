# Architecture revision: remove the memory barrier, then test a new model

The best measured model remains exp_002: validation PQ 0.4403668 and reported
public 0.36. No new accuracy result is claimed by this revision. The first new
implementation is exp_032, an exact chunked mask loss for YOLO11. Mask2Former is
the next architecture experiment if it passes memory and resolution probes.

## Correct the premises before choosing the architecture

The handoff is valuable experimental history, but several explanations are
hypotheses rather than measured constraints:

* **Mask2Former also constructs masks from a shared feature basis.** Its
  [official decoder](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py)
  takes a dot product of query embeddings and pixel features. The case for it is
  masked attention, richer pixel features, and set matching; it is not the
  elimination of a shared basis. Set prediction still needs an emission rule.
* **A 16 GB training failure is not proof that larger hardware is necessary.**
  The [YOLO loss](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/utils/loss.py)
  expands targets and logits for positive anchors, including multiple anchors
  assigned to one instance. Chunking before expansion and checkpointing each
  chunk attacks that allocation directly. Backbone and shared prototype memory
  remain, so a full-model GPU probe must still establish fit.
* **Two-decimal display does not hide every gain below 0.06 local PQ.** A public
  change smaller than 0.01 can cross a rounding boundary. Four rounded results
  cannot identify a precise conversion from local improvements to public ones.
  The observed 0.067–0.078 offset is descriptive evidence from five results,
  not a law or a reason to discard small validated improvements.
* **Pairwise human agreement is not a model ceiling.** SQ is conditional on
  matching, so pairwise SQ cannot establish a hard upper bound for a model with
  different matches. The failed union-labelled verifier proves that training
  target was mismatched; it does not prove that per-record probability of
  annotation cannot be learned. Equal year/station marginals also do not rule
  out all distribution shift. Keep these uncertainties explicit.
* **The stated grid arithmetic contains an error.** With the handoff's formula,
  `2048 * mask_ratio / imgsz`, ratio 4 at image size 2048 is 4 native pixels,
  not 2. Target resolution, prototype resolution, and information in the image
  are different. A polygon correction can affect coarse rasterization; its
  influence is not categorically zero below one grid cell.

The measured negative experiments remain useful. This revision does not repeat
post-hoc boundary correction, union-labelled reranking, or checkpoint fusion.

## Implemented: checkpointed positive-anchor mask loss

`shared/chunked_mask_loss.py` changes the execution of the mask objective:

1. Retain the compact instance-ID target map and the shared prototypes.
2. Select at most eight positive anchors per chunk, including repeated targets.
3. Expand only their targets, compute BCE with upstream cropping and normalization.
4. Save chunk inputs; recompute target/logit/BCE intermediates during backward.
5. Sum every contribution and divide once by the total positive-anchor count.

The large target/logit workspace scales with chunk size instead of all positive
anchors. A loop without checkpointing would keep the BCE graphs and would not
solve the backward memory problem. This implementation preserves the detection
head, weights, assignment, and segmentation objective. It can test full-resolution
supervision without simultaneously changing model capacity.

The exp_032 training entry point opts in explicitly, installs the criterion on
both training and EMA validation models, reads back critical settings, and writes
`resolved_experiment.json` beside checkpoints. It pins Ultralytics because it
uses an internal API. It fails on configuration drift and never falls back to a
coarser mask ratio. Historical experiment entry points remain reproducible.

Validation consists of upstream differential loss/gradient tests, mixed
precision, empty targets, retained-tensor checks, and a separate process memory
probe. These verify implementation and memory behavior, not segmentation quality.
`scripts/audit_resolution.py` uses all canonical training annotations to measure
crowding and nearest-neighbor round-trip damage, one COCO mask at a time.

The adapter also handles two edge cases in the pinned generic loss wrapper:
YOLO11's batch of two is a tensor, not a prototype/semantic tuple, and empty
batches have no semantic tensors. CPU cropping must preserve the upstream
image-level branch at 50 positive anchors; choosing it per chunk changes
fractional bounding-box rounding. Differential tests cover both branches.

### Results from this workspace

The [isolated CPU report](../results/exp_032_loss_probe_cpu.json) uses 256-square
maps, 32 prototypes, 40 synthetic instances and 400 positive anchors, chunk size
8, FP32, four CPU threads, and fresh processes. It measures two passes:

| Measurement | Upstream | Chunked |
|---|---:|---:|
| Retained autograd storage | 244,371,416 bytes | 8,716,360 bytes |
| Peak process working set | 978,354,176 bytes | 352,382,976 bytes |
| Warm forward/backward | 0.661 seconds | 1.262 seconds |
| Loss | 2.340498686 | 2.340498447 |

That is **28.0 times less retained storage and 64.0% lower peak process memory**,
at approximately 1.9 times the warm CPU execution time. These are isolated loss
measurements, not projected T4 training savings. CPU float32/bfloat16 loss and
gradient tests compare against the installed upstream implementation; float64
full-detector tests separate mathematical equivalence from reduction roundoff.
The actual CLI smoke test completes training, EMA validation, checkpoint saving,
reloading and inference on a tiny synthetic dataset. Its metrics have no
scientific interpretation.

Final local verification: `python -m pytest tests -q` completed with **75 passed**
on 2026-09-07. Six warnings originate in the existing pycocotools/NumPy decode
path; the new training subprocess completed without warnings. Compilation and
`git diff --check` also passed. Existing user edits to `CLAUDE.MD` and the deletion
of the root `HANDOFF.md` were preserved.

The [training-label audit](../results/exp_032_resolution_audit.json) covers all
6,874 instances in 974 records, preserving the canonical 601/106 photograph split.
Median crowding is seven instances, 95th percentile fifteen, maximum twenty-three.
No instances disappear at 800, 1024, 1280, 1536 or 2048. However, mean mask IoU
after nearest-neighbor down/up sampling is 0.684 at 800 and 0.879 at 1024.
The lower 0.778 at 1280 reflects grid alignment in this particular diagnostic,
**not evidence that 1024 model training is superior to 1280**. Report the
round-trip method explicitly; do not turn survival into a learned-quality claim.

## Next controlled experiments

| Stage | Fixed conditions | Decision evidence |
|---|---|---|
| Full-model fit probe | exp_002 checkpoint; corrected labels; ratio 1; batch 1; 2048; crowded training record | At least two optimizer steps (Adam state allocated), EMA validation loss, finite gradients, peak CUDA memory, step time |
| Matched control | Same seed/checkpoint, labels, augmentations, resolution and schedule; upstream versus chunked loss | Differential tests first; training behavior should agree within numerical tolerance |
| Native supervision | Full-resolution run with a completed cosine schedule; no backbone freeze by default | Canonical pooled PQ; resweep erosion/score/area; shared validation/submission rule |
| Mask2Former probe | Tiny/small COCO checkpoints; 800/1024/1280/1536; real train crowding, not just 15 masks | Isolated forward/backward/optimizer steps; memory, throughput, query count and resolution survival |
| Architecture trial | One best feasible Mask2Former configuration, COCO masks, one annotation record per sample | Same canonical validation photographs/records and native output scoring |

Measure epoch duration before setting the long-run horizon; 50 epochs is a
starting argument, not a promise it fits 12 hours. Leave time for validation and
checkpoint export. Do not start a long run until the complete training/validation
lifecycle fits. Upload finished checkpoints as durable datasets when using Kaggle.

For Mask2Former, use **one annotation record** per training item, never union
labels. Group photographs only for splitting and uncertainty estimates. Decode
with pycocotools, preserve empty records and instance identity, and choose a query
count above the measured crowding before training. Replicate grayscale to three
channels. Avoid mosaic and color changes unrelated to grayscale intensity.

Score every validation annotation record, pooling TP/FP/FN exactly as today.
Preserve seed 2026 and the 601/106 photograph split. Add paired bootstrap intervals
over photographs (all annotation records follow each sampled photograph) when
comparing new checkpoints; this is an additional report, not a replacement metric.
Tune on validation, retain the baseline operating point as a fixed comparison,
and reserve any additional robustness holdout for predeclared final comparisons.

## A genuinely different fallback: query-conditioned local mask decoding

If full-disk set prediction loses thin structures at the resolution that fits,
the more targeted redesign is a shared full-disk detector/backbone with a
**query-conditioned nonlinear mask decoder over native-scale regions**. Pass
each proposal's query, multiscale image features, relative coordinates and a
context halo to a small mask network; supervise it jointly with detector training
using COCO rasters. Decode regions in chunks and project directly into the native
frame. This changes mask representation instead of correcting finished masks.

Keep one instance identity across the entire region, including disconnected
parts. Long filaments may span most of the image: account for that worst case,
and use overlapping spatial windows with a common query if required. Train on
out-of-fold or online proposals to avoid the crop-domain mismatch of prior
refiners. Start with a frozen detector control, then joint training, and compare
against exp_032 under the same labels and compute allowance. This is a proposal,
not an implemented model or a predicted score gain.

## Scope of current evidence

Only two photographs and the training annotation JSON are present locally;
no trained checkpoint or CUDA device is available in this workspace. CPU
mathematical and lifecycle tests can be completed here. GPU fit and competition
PQ require the full images, baseline checkpoint and a GPU run. No new training,
public submission or paid compute should be inferred from these code changes.
