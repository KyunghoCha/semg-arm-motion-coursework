# Development amendment 2 — final candidate round

Before outer/test evaluation, amendment1 produced spectral256_multitask
accuracy47.3818%, macro-F1.430905 and spectral_gain_multitask46.0606%,
.430686. The original multitask had92.7451% pooled gesture accuracy but
51.3072% angle accuracy;712of823errors were the opposite angle of the
same gesture. The weighted control reached38.2121%/.341546.

Three final hypotheses, each with the same1s and the same five recording
folds, seeds, training-only normalization and existing trainer:

1. `spectral256_hierarchical`: shared 256-sample STFT CNN features; a
   4-gesture head and four conditional2-angle heads. The8class probability
   is P(gesture)×P(angle|gesture). Loss remains CE8 +.3 CEgesture
   +.3 CEmarginal_angle, with class weighting on CE8. This is a proposed
   factorization; no known gesture/participant is supplied at inference.
2. `spectral256_hierarchical_rms`: same factorization with5channel log10RMS
   values from the identical raw1s appended only to the angle head input.
   Training-fold mean/SD normalize those channels. Test/validation statistics
   are unused. The spectral encoder still receives only645spectral channels.
3. `spectral256_gain_multitask`: combine256STFT and the already specified
   uniform channel/shared log-amplitude gains from amendment1.

This round adds15fits and is the last candidate round. Rank all23single
models by development mean macro-F1. Form exactly one equally weighted
ensemble of the top two1sneural models under that ranking, and select it
only if mean macro-F1 improves on the best single. Freeze membership,
median best epochs and two final seeds42/2026 before opening any outer
or original test score. Primary nested evaluation remains the original
16-candidate procedure. No further tuning follows evaluation.
