# HANDOFF — state of the effort, 2026-09-01

Written to the protocol in `CLAUDE.MD` §6. Read this first, then
`docs/leaderboard-analysis.md` for the evidence behind every claim here.

## Project state

| | |
|---|---|
| Best submission | **0.32** public, rank ~227 / 467 |
| Best model | `cosmicwitness/sol-exp002-yolo-seg` output, `checkpoints/best.pt` |
| Leaderboard #1 | 0.55 (twelve teams, identical score) |
| Honest frontier | ~0.40 |
| Quota | GPU 4.66 h, TPU 16.9 h, resets 2026-09-05 |
| Submissions | 5/day; 4 remaining today |

## What has been tried

| exp | approach | training | LB |
|---|---|---|---|
| 001 | U-Net++ eff-b4 @512, semantic + connected components | 46 epochs, early stopped | 0.26 |
| 002 | **yolo11m-seg @1280, instance segmentation** | 149 epochs, **cut off still improving** | **0.32** |
| 003 | yolo11m-seg @2048, fine-tune of 002 | **never completed** | — |
| 004 | U-Net resnet34 @1024, TPU SPMD | 300 epochs, **converged** | 0.28 |

## What worked

**Direct instance prediction.** The single largest gain in the project, +0.06,
came from replacing semantic-segmentation-plus-connected-components with a model
that predicts instances directly. Recognition quality rose from 0.470 to 0.581
with segmentation quality unchanged, which is precisely where the gain was
predicted to appear.

**Confidence-ordered overlap painting.** The public 0.55 notebook sorts by
`boxes.cls`, which is identically zero under `nc=1`, so its panoptic painting
hands contested pixels to an arbitrary mask. `boxes.conf` is correct.

**TPU via SPMD.** `xmp.spawn` is broken on Kaggle's v5litepod-8, but a single
process sees all eight cores and SPMD shards across them: 300 converged epochs in
3.8 h at 1.4 min/epoch, against the T4's 3.4 min/epoch at 1280.

## What did not work

**The dense line is finished at ~0.28.** exp_004 removed both suspected
constraints — doubled resolution, trained to genuine convergence — and gained
+0.02 over exp_001, still losing to an undertrained instance model by 0.04.

**Every post-hoc decoder.** Morphological closing, spine seeding (+0.0005),
center/offset grouping (-0.166 at realistic noise). All measured before
spending compute.

**Threshold tuning.** The full sweep is worth +0.0022. The PQ surface is flat
because a false positive and a false negative both cost half a unit.

## Methodological warning — read before designing the next ablation

Three decoder ablations were run against **ground-truth** masks, where connected
components already score 0.9995. No decoder can demonstrate a gain there, so all
three were structurally incapable of answering the question asked of them. A
decoder question can only be settled on **predicted** masks, which costs a
training run. An earlier claim in this repository — that filaments were being
fragmented and triply penalised — was a plausible mechanism promoted to a finding
without test, and has been withdrawn.

## Next experiment, in priority order

**1. exp_003 — 2048 fine-tune of exp_002.** The only idea with positive evidence
behind it. Keeps the architecture that measurably wins and adds the resolution
the dense line could not exploit. exp_002's best epoch was its last, so it is
undertrained independently of resolution, and this run addresses both at once.
Needs ~6 h GPU. `kaggle kernels push -p kaggle/exp003`. Verify the log shows
`resuming from /kaggle/input/.../best.pt` rather than a fallback to COCO.

**2. Simply train exp_002 longer at 1280.** Lower variance than (1) and isolates
the epoch deficit from resolution. Same command with `IMGSZ` unchanged.

**3. Ensemble 002 and a second seed.** Only after (1) or (2), and only with
validation evidence — instance-level ensembling is easy to get wrong.

Do **not** spend a run on threshold tuning or a new decoder.

## Open questions

- Is the 0.40-0.55 gap crossable by modelling at all? Nothing between those
  values exists on a 467-team leaderboard, which is not what a difficulty cliff
  looks like.
- Would a large backbone at 2048 hold a usable batch on better hardware? On a
  14.56 GiB T4 it could only run one image per GPU, which is why exp_003 uses
  the medium backbone.

## Files to read

- `docs/leaderboard-analysis.md` — all evidence, including the leaderboard
  integrity finding and every negative result
- `docs/RESUME.md` — exact restart commands and what is banked where
- `experiments/exp_001_baseline/RESULTS.md` — the PQ decomposition that motivated
  everything after it
- `kaggle/exp003/` — the next run, configured and ready
