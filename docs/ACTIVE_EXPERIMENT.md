# Active objective and resource constraints

Latest user instruction: obtain a verified public leaderboard score **greater
than 0.46** for filament-segmentation-2026. Training, inference, postprocessing and
submission are authorized. Use the user's Kaggle CPU/TPU resources or other free
compute. **No GPU use and no further RunPod access or changes.** This supersedes
the prior frozen-weights-only restriction and RunPod rental request.

Verified current public best: 0.38 (Kaggle submission ref `56096948`, 2026-09-08).
Canonical exp_002 validation baseline: 0.4403668271. A code improvement or a local
PQ gain does not complete the objective; only the public submission score does.

The previous RunPod inference was never training. An attempt to cancel just that
process after the new restriction returned SSH connection refused. Do not retry
or change that instance. No current RunPod runtime state is asserted.

Current work: exp_033 frozen-weight CPU inference screen with a mandatory baseline
reproduction gate, plus an isolated TPU feasibility probe before any long run.
Preserve canonical seed/split 2026 and score individual annotation records.

2026-09-07 current runs:
- `cosmicwitness/sol-exp033-frozen-cpu`: RUNNING; identity/gamma 0.8/gamma 1.2.
- `cosmicwitness/sol-exp034-tpu-forward`: QUEUED; native exp002 TPU forward probe.
- `kaggle/exp035_mask2_probe`: prepared, push rejected because Kaggle permits
  only one batch TPU session. Not launched. Run after exp034 terminates.
- Local `experiments.exp_033_inference.src.cached_tta`: exp005 cache reproduces
  PQ 0.4403668270817509 exactly. Testing the confidence threshold omitted from
  the historical D4 fusion evaluation. First fused cache saved in
  `data/exp033_cached_tta`; resumed with equivalent prefix-paint evaluation.
- Quota API reports 72,000 TPU seconds allowed, 1,082.856 used (~19.7h left).
  GPU remains forbidden regardless of quota reporting.
- Fourteen exp033 tests pass, including prefix painting equivalence for
  overlapping masks, tied confidence, erosion and duplicate annotation records.

Public-method check: `lamhuy8904/solar-filament-unet-segmentation-0-55`
does not generate its submitted predictions with the displayed trained U-Net.
Its test loop copies an embedded compressed CSV (`CHAMPION_PAYLOAD`). The
payload's provenance is unverified; do not use it as model-quality evidence or
as our submission. Source inspected read-only under ignored `data/public_code`.

Measured progress:
- Corrected D4 fusion (link 0.5, vote 0.5, confidence 0.40, erosion 1px,
  min area 300) reaches local PQ **0.4570759555**. Public score remains 0.36.
- Sparse RLE voting reproduces all 1,414 first-rule fused masks byte-for-byte,
  including scores, in 1.73s. Remaining fusion settings are still being scored.
- CPU ONNX probe completed; warm timings are noisy and do not establish a speed
  gain. Do not switch deployment backend on this evidence. Existing PyTorch used.
- `cosmicwitness/sol-exp037-test-0` and `sol-exp037-test-1`, version 1, pushed:
  CPU only, 90 test photographs each, eight D4 views, source floor confidence
  **0.25** (matches historical exp007), native 2048, IoU 0.60, max_det 100.
  Both first compare three validation photographs with the historical cache;
  reject candidate-count, mask-IoU or confidence discrepancies. They only harvest
  candidates; final fusion configuration is selected from validation results.
- Local active sweep was restarted to use sparse voting and resume saved rows.
  Current terminal session: 27687. ONNX session 78193 finished successfully.

2026-09-08 update:
- Full corrected D4 sweep complete: best remains PQ 0.4570759555. A 20,000-draw
  photograph-cluster bootstrap estimates gain +0.01662, 95% interval
  [+0.00387, +0.03032], P(gain>0)=0.99515.
- CPU identity/intensity sweep completed. Current-runtime identity PQ 0.44106;
  gamma 0.8 PQ 0.42944 and gamma 1.2 PQ 0.43465. Intensity transforms rejected.
- TPU forward probe completed: warm eight-image 2048 forward 0.794s, finite,
  but raw outputs differ substantially from CPU. Do not use TPU predictions
  until end-to-end canonical PQ validates them.
- `sol-exp038-yolo-tpu-probe` is RUNNING: two real optimizer steps at 1024/mask2,
  1024/mask1, and 2048/mask1. It contains GPU-disabled metadata.
- CPU test shards `sol-exp037-test-{0,1}` v5 are RUNNING. Historical-cache gate
  accepts the verified version skew: identical candidate count, max confidence
  difference 2.1e-6, min mask IoU 0.9613 and mean IoU 0.9861 on the gate photo.
- User explicitly authorized new model training on TPU and TPU CSV generation.

2026-09-08 continuation after usage-limit interruption:
- `sol-exp037-test-{0,1}` v5 completed and both artifacts passed their historical
  cache gates. The selected D4 rule produced 1,110 disjoint instances for all 180
  test photographs. Submission ref `56096948` scored 0.38; local PQ 0.4570759555.
- `sol-exp038-yolo-tpu-probe` completed. All three configurations timed out during
  XLA initialization/compilation before an optimizer step. Do not launch the full
  YOLO TPU trainer without changing the graph/runtime approach.
- `sol-exp043-public-cpu` reproduced the imported public submission byte-for-byte
  (MD5 `09c0eae5fdff29a91f8a740475ecaf36`). The live public score for ref
  `56091487` is 0.34, disproving the artifact's stale advertised 0.70 score.
- `sol-exp039-tpu-validation` v1 was pushed to test end-to-end PQ for TPU inference
  from the existing exp002 weights. `exp035_mask2_probe` remains next because the
  account permits only one concurrent batch TPU session.

Continuation: user deleted queued exp039. exp035 v1 then ran, but tiny1024,
tiny1536 and small1024 all hit 900s timeout without two reported optimizer steps.
The logs do not establish the cause: tiny1536 reached loss scalar conversion.
exp035 v2 pushed: tiny256 then tiny1024, explicit synchronized stage timings,
three finite optimizer steps required, stop on first failed configuration.
Hidden monitor downloads v2 artifacts to data/exp035_output_v2 and ends for review.
Long exp046 training must not launch from this synthetic probe alone: its real-data
loss differs, schedule/checkpoint handling needs validation, and the prior 3300s
launch timeout contradicted its 10h budget. Monitor now requires an exp046-validated
marker after a matching real-data lifecycle check; that marker does not exist.
Fusion sweep completed: D4 control PQ .457076 remains best; best cross-model fusion
.450529 is rejected. Public best remains .38; no subsequent submission.

2026-09-08 training correction:
- Replaced exp046's unvalidated fixed-order query objective with CPU Hungarian
  assignment and fixed-shape class/BCE/Dice loss in shared/query_mask_loss.py.
  Four tests cover target/query permutation, gradient equivalence, inactive
  padding, empty records, nonfinite input and excess-instance rejection.
- New source kaggle/exp046_mask2_train/train.py completed three real-image CPU
  lifecycle steps using a small random 128px model, then saved/reloaded a checkpoint
  and ran a held-out photograph. This is code validation, not accuracy evidence.
- The trainer uses canonical split, one annotation record/sample, fractional area
  targets, measured step-budget cosine schedule and atomic checkpoint every100 steps.
- scripts/package_mask2_train.py embeds identical code for exp047 real TPU probe
  and exp046 training. Neither has launched yet.
- Replaced the PowerShell monitor body with launcher for scripts/monitor_training.py.
  It retains current exp035v2, downloads its diagnostics, launches exp047 only on
  finite 1024px completion, and launches exp046 only on matching-code real-data
  TPU probe success (3 finite steps, warm steps<30s, checkpoint reload+val forward).
  Three monitor tests prevent duplicate launches and restart on observation errors.
- Monitor logs/status: data/monitor/training-monitor.log and training-state.json.
  It automatically stops for review on failed probes and after training download.

exp048 CPU evaluation prepared and packaged. Uses identical preprocessing, native
mask resizing, query mask-quality scores and confidence painting for validation
and test. Canonical pooled PQ selects among 24 coarse rules; CSV generation only
runs when all 106 validation photographs complete and PQ exceeds .4570759555.
CPU lifecycle checkpoint smoke exercised inference/rasterization/scoring and
correctly refused submission generation. Monitor now advances finished exp046 to
exp048 and downloads its results, then stops for submission review or a new idea.
Kaggle quota checked through authenticated CLI: 7072.447 TPU seconds used of72000,
reserved0; exp035v2 remains QUEUED with failureMessage=null. MCP returned
Unauthenticated, so CLI remains the functioning connection. Queue cause unknown.
