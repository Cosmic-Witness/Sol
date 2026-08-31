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
