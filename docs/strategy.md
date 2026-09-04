# Strategy

Where the effort is going, why, and what would show it to be wrong.
Companion to `docs/leaderboard-analysis.md`, which holds the measurements this
argues from; every number here is transcribed from a file in `results/`.

**State at the time of writing.** Public 0.36, validation PQ 0.4404, rank about
92 of 502. Goal is a public score above 0.46. The leaderboard's top cluster sits
at 0.55-0.56.

---

## 1. The argument in one page

Ten experiments were run before the error was ever decomposed. Each targeted one
failure mode and each assumed the mode it targeted was the dominant one:
threshold tuning, spine seeding, disk masking, sub-pixel trimming, calibrated
emission, dihedral TTA, two refiners. Their combined contribution to the score is
approximately zero.

Decomposing the error first (exp_013, exp_014, exp_016) says why, and says what
is left:

| what the error is | share | reachable? |
|---|---|---|
| instances one side does not acknowledge exist | ~50% | **no** — annotators disagree at the same rate |
| mask on the right object but too wrong to match | 195 truths, 150 predictions | **yes** |
| covered by a candidate confidence discards | 305 truths | yes, but needs 3x the base rate to break even |
| invisible to the detector entirely | 2.1% | only with a different detector |
| split or merged instances | 7 of 936 | not a real category |

Then, separately: **the detector was never trained properly.** It was trained at
1280 and every submission since has inferred at 2048; its targets were rasterised
11% fatter than the scorer's convention; and it stopped because a `time=8.5`
budget expired at 149 epochs, not because it converged.

So the strategy is one sentence: **stop correcting the detector's output and
train the detector correctly.** That is exp_010.

---

## 2. Why the corrections were always going to fail

This is the part worth internalising, because it cost ten experiments to learn
and the reasoning generalises.

Panoptic Quality is `SQ x RQ` — mean IoU over matches, times a matched-instance
rate. Post-hoc geometry can only move a mask that already exists. It cannot
create an instance the detector never proposed, and it cannot remove label noise.
Half the error is one of those two, so **half the error was never addressable by
anything applied after inference.** Every corrector was competing for a share of
the other half, and the largest of them — the boundary — turns out to be capped
too:

| | PQ | SQ | RQ |
|---|---|---|---|
| annotator vs annotator | 0.3361 | 0.6348 | 0.5296 |
| **this detector** | **0.4404** | **0.6843** | **0.6436** |

The model already agrees with an annotator better than a second annotator does,
on both components. A corrector asked to improve on that has to predict *which
way one particular person resolved an ambiguity*, and there is no signal for that
in the image. The two refiners are the clean demonstration: the synthetic one
learned its task beautifully (IoU 0.7208 to 0.8529) and lost PQ; the real one had
the right task and could not learn it at all (0.6105 to 0.6335, reached at epoch
two, never beaten in 137 more).

**The corollary that matters for the oracles.** The "perfect masks are worth
+0.203" figure assumes SQ can reach 1.0 against a single annotator. It cannot.
Treating each annotator as a noisy draw around a latent boundary, a pairwise
disagreement of 0.365 of the union implies about half that per annotator, so a
model predicting the latent truth exactly scores about **SQ 0.82**. The rim
analysis reached 0.855 by a different route. The honest lever is 0.684 -> 0.82,
worth about +0.09, not +0.20.

---

## 3. The plan

**exp_010 — retrain the detector.** One run, three corrections, because they only
work together:

1. **Train at 2048.** Matches the inference resolution for the first time. The
   anchor-free head's response range over object sizes is fixed at training time,
   which is the most plausible reading of why inference at 2560 and 3072 degrade
   so sharply while 2048 gains.
2. **Correct the targets.** A 0.5px inward polygon buffer reconciles
   `cv2.fillPoly` with pycocotools: IoU 0.898 -> 0.959, area ratio 1.111 -> 0.986.
3. **`mask_ratio=1`.** Supervise the mask loss at 2048 instead of 512.

**Three and two are one change, not two.** At the default `mask_ratio=4` the loss
is computed on a 512 grid, where a half-pixel correction is an eighth of a pixel
— below what the grid can represent. Shipping the polygon offset without full
resolution would discard it before the model ever saw it. This is why the OOM
ladder concedes batch size before it concedes mask resolution.

Plus: no time budget, and checkpoints in the kernel output so Kaggle's 12-hour
cap is a checkpoint rather than a deadline.

**exp_015 — submit, separately, on CPU.** Training and submission were one kernel,
which meant training had to end early enough to leave room for inference or
nothing came out. Now exp_015 takes a submission from whatever checkpoint exists,
whenever one is wanted, and it runs on CPU because the CPU path is bit-for-bit the
GPU path (same MD5 on the CSV that scored 0.36). Ninety free minutes buys what
would otherwise be twenty minutes of the scarce resource.

### The arithmetic, and why the target is 0.56 rather than 0.54

Both submissions with a measured validation score show the same offset: 0.4064
against public 0.33, and 0.4404 against 0.36. Two gaps of 0.078. Naively, public
0.46 needs validation 0.54.

That offset is not established. Covariate shift is excluded (train and test agree
by year to a couple of points and by station to within three) and so is the
weighting (counting each photograph once instead of once per annotation moves the
figure by 0.0008). What remains is ordinary generalisation plus the sampling noise
of a public leaderboard computed on a fraction of the test set, and two
submissions cannot separate them. **So aim above what the arithmetic asks:
validation 0.56.**

Is that reachable? Taking SQ from 0.684 to 0.76 — just over half the way to the
0.82 ceiling — while carrying two thirds of the 150 near misses over the matching
threshold gives validation 0.528. The rest has to come from recall, and a
detector trained at the inference resolution is the one thing that plausibly
supplies it: 305 truths already have a covering candidate that confidence rates
too low, some of which are small filaments a 1280-trained model is unsure about.

**This is the honest position: the arithmetic reaches roughly 0.53 and the target
is 0.56.** The gap is carried by recall improvements that are argued for rather
than measured. exp_010 may well land short.

---

## 4. What is deliberately not being done

| | why |
|---|---|
| any further post-inference correction | three refiners and six geometric methods; the mechanism is understood and it is exhausted |
| chasing the orphan class | it is the label noise floor |
| test-time augmentation | lost 0.037, and the model is already flip-augmented so flips are near-redundant |
| consensus targets | two annotators averaged predict a third no better than one does (+0.0004) |
| dense semantic models | 0.26 and 0.28 against the instance model's 0.32 |
| **using the idle TPU** | the refiner was the only TPU-shaped work. Free compute is not a reason to build a fourth corrector |
| `copy_paste` augmentation, larger backbones | plausible, but untestable inside one 30-hour shot; one change at a time when each run costs the week's quota |

The last row is the discipline this project has most often failed at. exp_003
burned six hours on a configuration that was never verified as deployed; a paid
run lost $2.39 to a pod with no persistent volume. **Verify the thing is running
what you think before it runs, not after.** Three bugs in exp_010 were caught this
way and any one would have wasted the full thirty hours:

- `optimizer='auto'` logs `ignoring 'lr0='` and picks MuSGD at 0.01 — a
  from-scratch rate applied to a fine-tune.
- A bare `resume=True` routes to `get_latest_run()`, which searches Ultralytics'
  own settings directory, so the 12-hour-cap resume would have resumed nothing.
- The OOM ladder conceded mask resolution, which would have silently made the
  polygon offset inert.

---

## 5. What would falsify this, and what happens then

**The claim is that the detector is undertrained, not that it is
architecturally limited.** It is falsified if exp_010 converges — properly, at
2048, on correct targets, with patience 60 rather than a clock — and validation
does not move materially past 0.4404.

If that happens, the reading is that yolo11m-seg's 32-prototype basis at stride 4
is the binding constraint, and the responses in order of cost:

1. **Weight the mask loss near the target boundary.** Full-resolution supervision
   down-weights the rim about fourfold relative to the interior (a 1275-pixel
   filament has 15.7% of its mask on the rim at 2048, against 62% at 512). That is
   the right trade against a rim that is not representable at all, but it is a
   trade, and it is recoverable with a weighted loss. One more training run.
2. **Raise `nm` above 32.** Directly addresses the prototype basis. Costs the
   pretrained mask head, which on 601 photographs is a real cost.
3. **A dense semantic model at 2048, fused into YOLO's instances.** The one
   remaining idea that supplies independent evidence rather than a learned
   correction of YOLO's error — which is why it is listed here and not in section
   4. Connected components on ground-truth masks score PQ 0.9995, so instance
   separation is not the obstacle it appeared to be.

**Secondary lever, if a cheap one is wanted:** the 305 truths that are seen and
disbelieved. Promoting *k* right and *m* wrong candidates from the discarded band
moves PQ to `(578.2 + 0.65k) / (1313 + 0.5k + 0.5m)`, so break-even is two wrong
per one right — precision above 33% against a 10.5% base rate. exp_005's
gradient-boosted model over four geometric features could not clear it, and its
verdict was that the features "add nothing beyond what confidence already
encodes". exp_019 tries a quantity confidence genuinely does not see: confidence
scores the *box*, while how fast the mask field falls away from its cut says
whether the mask is a filament holding a strongly positive interior or a smear
that barely crosses zero.
