# Raw results

The JSON each kernel wrote, unedited. Every table in `docs/leaderboard-analysis.md`
and every number in `docs/strategy.md` is transcribed from a file here, so a
reader can check the transcription without a Kaggle account and without re-running
anything.

| file | produced by | what it holds |
|---|---|---|
| `exp_005_nearmiss_sweep.json` | `sol-exp005-nearmiss` | the operating-point grid that settled conf 0.35 / min_area 300 / grow -1 at PQ 0.4403668 |
| `exp_009_refiner_synthetic_val.json` | `sol-exp009-refine-eval` | the synthetic-damage refiner against the detector |
| `exp_012_refiner_real_training.log` | `sol-exp012-refiner-real` | per-epoch training curve of the real-pair refiner |
| `exp_013_decomposition.json` | `sol-exp013-errors` | the failure taxonomy and the oracle for eliminating each class |
| `exp_014_irreducible.json` | `sol-exp014-annotator` | detector-versus-annotator orphan rates against annotator-versus-annotator |
| `exp_015_nearmiss_sweep.json` | `sol-exp015-submit` | the same grid recomputed on CPU; identical to exp_005's, which is the point |
| `exp_016_recall_ceiling.json` | `sol-exp016-recall` | how much ground truth the candidate pool covers before confidence filters it |
| `exp_018_refiner_real_val.json` | `sol-exp018-refine-real-eval` | the real-pair refiner against the detector |
| `exp_020_multiplicity.json` | `sol-exp020-multiplicity` | validation PQ split by how many people annotated each photograph |

Two checks are worth running against these files rather than taking on trust.

**The decomposition reproduces the shipped score.** `exp_013_decomposition.json`
has `oracles.measured.pq = 0.4403668270817509`, and `exp_005_nearmiss_sweep.json`
has the same value in the row `conf 0.35 / min_area 300 / grow -1`. The analysis
is therefore accounting for the configuration that was actually submitted, not a
neighbouring one.

**The CPU and GPU inference paths agree exactly.** `exp_015_nearmiss_sweep.json`
was produced on a CPU kernel and `exp_005_nearmiss_sweep.json` on a T4; the
submission CSVs they wrote have the same MD5.

Model checkpoints are not here. They are tens of megabytes and live in the Kaggle
kernel outputs the table names.
