# Resume point — 2026-08-31

Training was cancelled mid-run to release GPU for higher-priority work.

## Banked

| | |
|---|---|
| Best submission | **0.32** public, rank ~227/467 (exp_002) |
| Weights | `cosmicwitness/sol-exp002-yolo-seg` output, `checkpoints/best.pt` (45 MB) |
| Validation PQ | 0.3736 at conf 0.30 / min_area 300 |
| Quota left | 7.75 h GPU, 20 h TPU — resets 2026-09-05 |
| Submissions | 3 of 5 remaining today |

The exp_002 checkpoint is durable on Kaggle and attaches to any kernel as a
`kernel_source`. Nothing needed for the resume is on local disk.

## Lost

exp_003's 1.5 h of 2048 fine-tuning. Ultralytics wrote its checkpoints under
`/kaggle/temp`, which is not part of the kernel output, so cancelling discarded
them. Fixed: `RUNS_DIR` now points at `/kaggle/working`, so a future run is
recoverable at whatever epoch it was interrupted.

## To resume

```
kaggle kernels push -p kaggle/exp003     # fine-tunes exp_002 at 2048
```

`kaggle/exp003/kernel-metadata.json` still carries the real config — GPU on,
T4, exp_002 attached as a kernel source. **The kernel currently sitting at that
slug on Kaggle is a CPU no-op stub**, pushed only to terminate the running
session; re-pushing from this repo restores the real driver.

Verify on the next run that `find_start_weights()` reports
`resuming from /kaggle/input/.../best.pt` rather than falling back to COCO.

## Open state

- Threshold tuning is exhausted (+0.0022) — do not spend a run on it.
- Recall is the wall: ~40% of filaments undetected at 1280 at any threshold.
  Resolution is the untested lever.
- Realistic ceiling for this approach is ~0.40. The 0.55 cluster is not a
  modelling result — see `docs/leaderboard-analysis.md`.
