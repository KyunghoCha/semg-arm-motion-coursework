# Local continuation: generalization v2

Frozen before any new model scores, 2026-10-20 UTC.
Parent baseline: 264f38a38246308235b30fbe77a0be7bb21f54cd.
Source lives on research/generalization-v2; generated/private outputs live in
the sibling step2_improvement_v2 directory. Original main and all old outputs
are preserved. Raw signals, lectures and weights remain local.

## Goal and evidence boundary

Improve eight-class recognition, principally cross-recording generalization.
The published A 50.51% trial result and B 97.14% trial vote result answer
different questions. Neither establishes paper replication. The publisher's
abstract confirms RF 96.72 ± 0.58% five-fold accuracy but not the detailed split,
segment start, aggregation or scaler fitting. The full paper is unavailable.
Lecture material is a secondary description of that paper and a primary
source for classroom instructions.

The original A/B tests have been seen repeatedly. New use of them is
exploratory. Do not claim any old participant or recording is untouched.
Only A development (51 records, 1530 trials) is used for new candidate decisions.
No A/B holdout score is computed until all selection files are frozen.

## Lecture-to-implementation mapping

| Item | Evidence in supplied slides | Current / continuation choice |
|---|---|---|
| units | week1 p7,11; main p23–24 | 64 CSV records each contain 30 trials, unlike example one-trial files |
| signal | main p19–25; week2 p9 | 1000 Hz, five channels; source already filtered, no second filter |
| segment | main p23,26; week2 p10 | classroom 2 s from 1.5 s is an assumption; old A/B use energy-selected 1 s |
| windows | week2 p11; main p26 | 200 ms / 100 ms hop: 19 for 2 s, 9 for 1 s |
| features | week2 p12; main p27–28 | RMS/MAV/WL/ZC(0.01)/SSC, 25 dimensions, feature-major channel order |
| normalization | week2 p7,14; main p30 | train-fold only; paper's exact fit order unknown |
| models | week3 p9; main p32–34 | SVM RBF C1 gamma scale; RF100 seed42; XGB400 depth6 lr.05 sub.9 col.8 if already installed |
| splitting | week2 p13; week3 p12; main p24,35,51 | stratified random trials 80/20 and five folds; overlapping windows stay together |
| measures | week3 p10–14 | accuracy, macro precision/recall/F1, fold SD, confusion, fit and 100 single-inference timings |
| scope | main p51,53–54 | random-trial performance is within dataset; LOSO is a distinct extension |

## Comparisons and candidates

1. **A development screening**: reuse the original five recording folds, and
report all candidates' trial accuracy and macro-F1. This is a development
comparison, including existing known validation results; not a fresh test.
2. **Nested recording CV**: retain those five outer recording folds. Inside
each outer train set, construct three stratified recording folds with seed
20261021 + outer index. Fit scaler, select candidate and early-stopping epoch
only inside inner folds. Refit selected model on the outer train set for the
median inner best epoch. Freeze all five choices before computing outer
scores. Compare with the original spectral1s architecture, whose epoch is
also determined by inner folds. The predefined family is reused in every
outer fold, not pruned using screening results. This limits model-selection
leakage within the run but cannot undo prior dataset-level design exposure.
3. **Final A exploratory comparison**: select by full development five-fold
macro-F1, refit with fixed seeds 42 and 2026, average neural probabilities.
Only after all choices/epochs are locked, evaluate the original 13-record
test once. Report the same split and changes of input duration separately.
4. **C classroom baseline**: separate protocol, all 1920 trials, stratified
80/20 seed42 and independent five-fold stratified trials seed42. Use fixed
1.5–3.5 s, 19 windows ×25 features, no log transform, fold-specific scaler.
Window predictions and majority trial predictions (smallest ID tie) are
reported separately. Never select the best split seed or call C A improvement.

Predefined CPU family: old 1s mean+SD 70 features SVM C1; 1s relative spectral
shape plus log-RMS SVM C1/C10; 5s log-RMS envelope SVM C1/C10; sample-relative
5s envelope SVM C1/C10 and shrinkage LDA; 2s energy-selected 25-feature mean+SD
SVM C1. No participant/record ID is a model input.

Predefined GPU family: original spectral1s CNN; spectral2s CNN;
5s envelope CNN with absolute, sample-relative or concatenated representations;
concatenated-envelope multitask CNN and 1s spectral multitask CNN. Multitask
loss is eight-class CE + 0.3 gesture CE + 0.3 angle CE; prediction remains
the eight-class head. Small envelope CNN preserves coarse temporal position.
All neural runs use CUDA, no CPU fallback.

Energy2s starts are chosen from 0.5–2.5 s at 0.1 s increments using this
trial's summed squared signal, earliest tie. Whole-trial envelope uses
50 nonoverlapping 100 ms RMS bins from marker to marker+5 s. Relative
envelope subtracts each channel's within-trial 20th-percentile log RMS.
This is a low-amplitude reference, not a verified physiological rest interval.
These sample-only transforms do not fit to validation records. Their 5s
context changes the information and latency budget and is reported explicitly.
Preparation inspection before any training found six 4,999-sample marker
intervals. For the fixed 5,000-sample envelope only, repeat the last sample
once inside that trial; never read the next trial. Longer intervals are
cropped at 5,000 samples. This boundary handling does not affect 1s/2s inputs.

CPU shape representation uses 1s channel-normalized periodogram band fractions
plus channel log RMS and channel correlations. No target-domain calibration.
SVM uses StandardScaler fit on training rows only and gamma=scale; LDA uses
lsqr with shrinkage 0.5. CPU SVM C values are candidates, not test-tuned values.

## Execution and decision rules

Accuracy is secondary; mean eight-class macro-F1 selects candidates.
Tie: lexicographic name. Neural settings: AdamW lr.001 weight_decay.001,
batch128, dropout.25, label smoothing.05, grad clip5. Original spectral
baseline retains max40/min8/patience8. Other candidates max60/min12/patience12.
Class weighting is inverse train-class frequency for new neural models only;
the old baseline remains unweighted. CV seed42+fold, nested seed4242+10*outer+inner.
Final epochs are rounded median best epochs, never chosen from outer/test.

GPU <=8GiB allocated, temperature <85 C, free disk >=5GiB, <=4 CPU threads,
model <=250k parameters, initial GPU fit budget90minutes. Resource check each
epoch. Do not terminate other processes. No installations, paid compute,
cloud substitution, remote publication, or LMS submission in this run.
Save logs/checkpoints, split indices, source/config/data hashes and every
candidate outcome. Abort on nonfinite values, overlap/hash/label mismatch.
OOM may retry once at batch64 with the failure retained.

After screening, use only development evidence to assess a second bounded
family if an identifiable failure remains; write an amendment before running
it. Such adaptive results must be labeled development/exploratory and excluded
from the already-frozen primary nested comparison. Stop adding candidates
after two such rounds or the resource budget. Do not use the final A test to
propose or rank follow-up models.

## Research basis and uncertainties

- Publisher abstract opened: https://link.springer.com/article/10.1007/s13534-026-00607-7
- Nested CV selection/outer split definition opened:
  https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Existing model and normalization/epoch/selection code inspected locally.
- Sample-relative envelope and auxiliary heads are proposed hypotheses,
  not claims of an established optimum or a reproduction of a published model.
- Source filtering boundaries, marker physiological semantics, electrode
  changes and acquisition chronology remain unknown. Eight participants and
  one recording per person-class limit causal attribution and generalization.
