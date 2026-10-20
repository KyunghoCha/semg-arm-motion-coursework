"""Recompute metrics and replay saved models without fitting or reading old test data."""
import json, time
import numpy as np
from sklearn.metrics import f1_score
from run_experiment import *


def main():
    freeze()
    s = dict(np.load(DATA / "splits.npz"))
    old_split = dict(np.load(TASK / "step2_improvement_v2/data/splits.npz"))
    assert np.array_equal(s["source_indices"], np.flatnonzero(old_split["development"]))
    assert not old_split["test"][s["source_indices"]].any()
    lock = json.loads((OUT / "results/SELECTION_LOCK.json").read_text())
    frozen_time = json.loads((OUT / "results/CONFIG_FROZEN.json").read_text())["timestamp"]
    checks = []
    count = 0
    for stage in ("screen", "inner", "outer", "diagnostics"):
        for name, cfg, label, tr, va in cases(stage, s):
            assert_split(s, tr, va)
            folder = OUT / "results" / label
            row = json.loads((folder / "summary.json").read_text())
            a = dict(np.load(folder / "predictions.npz"))
            assert row["config"] == cfg and row["name"] == name and row["device"] == "cuda:0"
            assert frozen_time < row["timestamp"]
            assert np.array_equal(a["indices"], np.flatnonzero(va))
            assert np.array_equal(a["true"], s["y"][va])
            assert np.array_equal(a["pred"], a["probabilities"].argmax(1))
            assert np.allclose(a["probabilities"].sum(1), 1, atol=2e-6)
            recalculated = score(a["true"], a["pred"])
            for key, value in recalculated.items():
                assert np.allclose(row[key], value), (label, key)
            assert np.isclose(row["macro_f1"], f1_score(a["true"], a["pred"], labels=range(8), average="macro", zero_division=0))
            if stage in ("outer", "diagnostics"):
                assert lock["timestamp"] < row["timestamp"]
            if stage == "outer":
                m = Model(cfg)
                for file in sorted(folder.glob("head*.ubj")):
                    head = xgb.Booster(model_file=file)
                    head.set_param({"device": "cuda:0", "nthread": 2})
                    m.heads.append(head)
                x = np.load(DATA / f"{cfg['feature']}.npy").astype(np.float32)[va]
                prob = m.predict_proba(x)
                assert np.array_equal(prob, a["probabilities"])
                t = time.monotonic()
                for _ in range(100):
                    m.predict_proba(x[:1])
                checks.append(dict(folder=label, exact_cuda_probability_replay=True,
                                   probability_max_abs_error=float(np.max(abs(prob-a["probabilities"]))),
                                   single_feature_input_ms=(time.monotonic()-t)*10,
                                   timing_includes_DMatrix_and_transfer=True, excludes_feature_extraction=True))
            count += 1
    assert count == 503
    assert len(list((OUT / "results").glob("**/summary.json"))) == 504
    write(OUT / "reports/verification.json", dict(timestamp=stamp(), evaluated_fits=count,
          representative_fit=1, development_only=True, previous_test_rows=0,
          all_metrics_recomputed=True, all_splits_disjoint=True, selection_precedes_outer=True,
          outer_checkpoint_replay=checks, features_and_source_hashes_verified=True))
    print("VERIFIED", count, "evaluations;", len(checks), "saved model replays", flush=True)


if __name__ == "__main__":
    main()
