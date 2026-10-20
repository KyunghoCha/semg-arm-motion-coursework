# Development amendment 1

Written after the initial 16-candidate development screen and inner CV, before
any nested outer or reused A test predictions. Initial spectral multitask
mean trial accuracy is 46.515%, macro-F1 0.421067; the original spectral
baseline gives 37.879%, 0.333765. The primary nested family and its locked
choices stay unchanged.

Four additional candidates use exactly the original energy-selected 1s:

- `spectral_weighted`: original architecture with the new family's
  inverse-class weighting and 60/min12/patience12 stopping; no auxiliary
  losses. This separates the multitask package from a weighted control.
- `spectral128_multitask`: 128-sample Hann STFT, hop16, no padding.
- `spectral256_multitask`: 256-sample Hann STFT, hop32, no padding.
- `spectral_gain_multitask`: original 64-sample STFT with training-only
  channel and shared amplitude gain. Each log amplitude gain is uniform
  [-0.15, 0.15] in natural units. Twice log10(gain) is added to log power;
  gain is constant across the sample's time/frequency bins. Validation and
  inference use no augmentation.

All use the existing frozen trainer, normalization from training rows,
seeds42+fold, original five recording folds, eight-class output and unchanged
optimizer. Multitask loss remains CE8 + .3 CEgesture + .3 CEangle.
No target-record calibration, participant input, or learned test statistics.
These are proposed hypotheses, not established optimal configurations.

Rank all single candidates by five-fold mean macro-F1, lexical tie. Also
evaluate one equal-probability ensemble of the top two 1s neural candidates
ranked by that same development criterion. No weight optimization. Select
the ensemble only if its mean macro-F1 strictly exceeds the best single
candidate. A final neural component averages seeds42 and2026, each fitted
for its rounded median development best epoch. All configurations, epochs
and ensemble membership must be locked before any outer/test evaluation.

This is adaptive development and is excluded from the primary nested
comparison. Primary nested outer results will estimate only the original
fixed16-candidate procedure, with the limitations of prior dataset exposure.
No outer/test score may guide another amendment. At most one further
development amendment is allowed before evaluation, only if these results
identify a specific actionable failure. Stop at the original90minute GPU
budget or resource limits. This amendment adds20development fits.
