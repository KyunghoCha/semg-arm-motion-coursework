"""Post-v4 adaptive development-only head-round comparison; no outer/test run."""
import json, sys, time, traceback
import numpy as np
from run_experiment import *

CASES = {f"{feature}_angle{rounds}": dict(feature=feature, structure="independent", depth=2, angle_rounds=rounds)
         for feature in ("legacy", "robust", "fusion", "relative") for rounds in (10, 25, 50, 100)}


def frozen():
    freeze()
    spec = dict(config=CASES, source_sha256=sha(__file__), plan_sha256=sha(HERE / "ANGLE_ROUNDS_PLAN.md"),
                previous_selection_sha256=sha(OUT / "results/SELECTION_LOCK.json"),
                input_manifest_sha256=sha(DATA / "manifest.json"),
                prior_outer_results_seen=True, purpose="adaptive development selection only; no repeated outer/test evaluation")
    p = OUT / "results/ANGLE_ROUNDS_FROZEN.json"
    if p.exists(): assert json.loads(p.read_text())["spec"] == spec
    else: write(p, dict(timestamp=stamp(), spec=spec))


class AngleModel(Model):
    def fit(self, x, y):
        for i, (target, classes, rounds) in enumerate([(y // 2, 4, 200), (y % 2, 2, self.config["angle_rounds"])]):
            counts = np.bincount(target, minlength=classes)
            assert (counts > 0).all()
            d = xgb.DMatrix(x, label=target, weight=len(y)/(classes*counts[target]), nthread=2)
            params = {**PARAMS, "max_depth": 2, "num_class": classes, "seed": PARAMS["seed"]+i}
            model = xgb.train(params, d, num_boost_round=rounds, verbose_eval=False)
            assert json.loads(model.save_config())["learner"]["generic_param"]["device"] == "cuda:0"
            self.heads.append(model)
        return self


def main():
    frozen()
    s = dict(np.load(DATA / "splits.npz"))
    start = time.monotonic()
    for name, cfg in CASES.items():
        x = np.load(DATA / f"{cfg['feature']}.npy").astype(np.float32)
        for k in range(5):
            folder = OUT / f"angle_rounds/{name}/fold{k}"
            if (folder / "summary.json").exists(): continue
            try:
                assert time.monotonic()-start < 900
                tr, va = s["fold"] != k, s["fold"] == k
                assert_split(s, tr, va); before = resource()
                t = time.monotonic(); model = AngleModel(cfg).fit(x[tr], s["y"][tr]); seconds = time.monotonic()-t
                prob = model.predict_proba(x[va]); pred = prob.argmax(1)
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=2e-6)
                row = {**score(s["y"][va], pred), "name": name, "config": cfg, "timestamp": stamp(),
                       "fit_seconds": seconds, "train_accuracy": float(np.mean(model.predict_proba(x[tr]).argmax(1)==s["y"][tr])),
                       "resources": [before, resource()], "device": "cuda:0"}
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "predictions.npz", indices=np.flatnonzero(va), true=s["y"][va], pred=pred, probabilities=prob)
                write(folder / "summary.json", row)
            except Exception:
                write(folder / "failure.json", dict(timestamp=stamp(), traceback=traceback.format_exc()))
                raise
            print(stamp(), name, k, f"acc={row['accuracy']:.5f} f1={row['macro_f1']:.5f} seconds={seconds:.2f}", flush=True)


if __name__ == "__main__": main()
