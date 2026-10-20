"""Inspect frozen source/data, preprocessing invariants and saved outer estimators."""
import json, sys, time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import joblib
from prepare import covlog, correlations
from train import *


def initial():
    freeze()
    manifest = json.loads((DATA/"manifest.json").read_text())
    for name, digest in manifest["hashes"].items():
        assert sha(DATA/name) == digest
    s = dict(np.load(DATA/"splits.npz"))
    old = dict(np.load(TASK/"step2_improvement_v2/data/splits.npz"))
    assert np.array_equal(s["source_indices"], np.flatnonzero(old["development"]))
    assert not np.any(old["test"][s["source_indices"]])
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(12, 5, 1000))
    a, b = covlog(raw, True), covlog(raw*7, True)
    assert np.allclose(a, b, atol=1e-12)
    assert np.allclose(correlations(raw), correlations(raw*np.arange(1, 6)[None, :, None]), atol=1e-12)
    unchanged = []
    for k in range(5):
        a = np.load(OUT/f"results/screen/legacy_svm1/fold{k}/predictions.npz")
        b = np.load(TASK/f"step2_improvement_v2/results/screen/legacy_svm1/fold{k}/validation.npz")
        assert np.array_equal(a["source_indices"], b["indices"])
        assert np.array_equal(a["pred"], b["pred"])
        unchanged.append(k)
    result = {"cached_features_sha256_verified": True, "development_only": True,
              "trace_normalized_covariance_gain_invariant": True,
              "channel_correlation_positive_gain_invariant": True,
              "legacy_svm_prediction_replay_exact_folds": unchanged}
    write(OUT/"reports/initial_verification.json", result)
    print(result, flush=True)


def outer():
    freeze()
    s = dict(np.load(DATA/"splits.npz")); checks = []
    lock = json.loads((OUT/"results/SELECTION_LOCK.json").read_text())
    for path in sorted((OUT/"results/outer").glob("outer*/*/summary.json")):
        row = json.loads(path.read_text())
        k = int(path.parent.parent.name.removeprefix("outer"))
        tr, va = s["fold"] != k, s["fold"] == k
        assert_split(s, tr, va)
        assert lock["timestamp"] < row["timestamp"]
        arr = np.load(path.parent/"predictions.npz")
        assert np.array_equal(arr["indices"], np.flatnonzero(va))
        assert np.array_equal(arr["true"], s["y"][va])
        assert np.isclose(accuracy_score(arr["true"], arr["pred"]), row["accuracy"])
        assert np.isclose(f1_score(arr["true"], arr["pred"], labels=range(8), average="macro", zero_division=0), row["macro_f1"])
        model = joblib.load(path.parent/"model.joblib")
        x = np.load(DATA/f"{row['config']['feature']}.npy")
        assert np.array_equal(model.scaler.mean_, x[tr].mean(0))
        assert np.array_equal(model.predict(x[va]), arr["pred"])
        replay_prob = model.predict_proba(x[va])
        probability_error = float(np.max(np.abs(replay_prob-arr["probabilities"])))
        assert np.allclose(replay_prob, arr["probabilities"], rtol=0, atol=1e-12)
        t = time.monotonic()
        for _ in range(100):
            model.predict(x[va][:1])
        checks.append({"folder": str(path.parent.relative_to(OUT)), "replay_exact": True,
                       "probability_max_abs_error": probability_error,
                       "train_only_scaler_exact": True, "single_feature_input_ms": (time.monotonic()-t)*10,
                       "excludes_feature_extraction": True})
    assert checks
    write(OUT/"reports/outer_verification.json", checks)
    print("OUTER CHECKPOINTS VERIFIED", len(checks), flush=True)


if __name__ == "__main__":
    {"initial": initial, "outer": outer}[sys.argv[1]]()
