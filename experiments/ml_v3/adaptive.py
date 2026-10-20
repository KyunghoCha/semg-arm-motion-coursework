"""Factor comparisons prompted by the completed57-candidate development screen."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time, json, traceback
import numpy as np
from sklearn.preprocessing import StandardScaler
from models import estimator
from train import OUT, DATA, HERE, TASK, CONFIG, score, stamp, sha, write, freeze, assert_split


def configs():
    c = {}
    def add(name, gf, af, gm="et", am="et", leaf=3, balanced=True, context="1s"):
        c[name] = {"gesture_feature": gf, "angle_feature": af, "gesture_model": gm, "angle_model": am,
                   "leaf": leaf, "balanced": balanced, "context": context}
    for f in ["legacy", "robust", "spectrum", "relations", "relative"]:
        add(f"{f}_ind_et3", f, f)
    for f in ["legacy", "robust", "spectrum", "relations", "fusion", "relative"]:
        add(f"{f}_ind_lda", f, f, "lda", "lda")
    for f in ["legacy", "fusion"]:
        add(f"{f}_ind_svm1", f, f, "svm", "svm")
    add("fusion_ind_et6", "fusion", "fusion", leaf=6)
    add("fusion_ind_et3_unweighted", "fusion", "fusion", balanced=False)
    add("fusion_ind_rf3", "fusion", "fusion", "rf", "rf")
    for af in ["logrms5", "logrms_upper", "logrms_upper_ratio", "relative_rms5"]:
        add(f"mixed_et_lda_{af}", "legacy", af, "et", "lda")
    for am in ["lda", "et"]:
        add(f"mixed_et_{am}_lowref5s", "legacy", "lowref5s", "et", am, context="5s")
    assert len(c) == 22
    return c


CANDIDATES = configs()


def prepare():
    freeze()
    assert len(list((OUT/"results/screen").glob("*/fold*/summary.json"))) == 285
    assert not (OUT/"results/SELECTION_LOCK.json").exists()
    path = DATA/"adaptive_features.npz"
    assert not path.exists()
    s = dict(np.load(DATA/"splits.npz"))
    # The source1s cache was prepared earlier; retain only original development indices.
    raw = np.load(TASK/"step2_gpu_pc/data/raw1000.npy")[s["source_indices"]].astype("float64")
    logrms = np.log10(np.sqrt(np.mean(raw*raw, -1)).clip(1e-12))
    env = np.load(TASK/"step2_improvement_v2/data/envelope_abs.npy")[s["source_indices"]]
    baseline = np.quantile(env, .2, axis=-1)
    np.savez(path, logrms5=logrms, logrms_upper=logrms[:, 3:],
             logrms_upper_ratio=np.c_[logrms[:, 3:], logrms[:, 3]-logrms[:, 4]],
             relative_rms5=logrms-logrms.mean(1, keepdims=True), lowref5s=logrms-baseline)
    write(OUT/"results/ADAPTIVE_FROZEN.json", {
        "timestamp": stamp(), "candidates": CANDIDATES, "source_sha256": sha(__file__),
        "plan_sha256": sha(HERE/"AMENDMENT_1.md"), "features_sha256": sha(path),
        "base_freeze_sha256": sha(OUT/"results/CONFIG_FROZEN.json"),
        "source_env_sha256": sha(TASK/"step2_improvement_v2/data/envelope_abs.npy")})
    print("ADAPTIVE PREPARED", len(CANDIDATES), flush=True)


def frozen():
    freeze()
    f = json.loads((OUT/"results/ADAPTIVE_FROZEN.json").read_text())
    assert f["candidates"] == CANDIDATES and f["source_sha256"] == sha(__file__)
    assert f["plan_sha256"] == sha(HERE/"AMENDMENT_1.md")
    assert f["features_sha256"] == sha(DATA/"adaptive_features.npz")


def base_config(kind, c):
    return {"model": kind, "leaf": c["leaf"], "balanced": c["balanced"], "C": 1}


def probabilities(model, x, kind):
    if kind != "svm":
        return model.predict_proba(x)
    z = model.decision_function(x)
    if z.ndim == 1:
        z = np.c_[-z/2, z/2]
    q = np.exp(z-z.max(1, keepdims=True))
    return q/q.sum(1, keepdims=True)


def run(partition):
    frozen()
    s = dict(np.load(DATA/"splits.npz")); y = s["y"]
    arrays = {k: np.load(DATA/f"{k}.npy") for k in ["legacy", "robust", "spectrum", "relations", "fusion", "relative"]}
    arrays.update(dict(np.load(DATA/"adaptive_features.npz")))
    start = time.monotonic()
    for i, (name, c) in enumerate(CANDIDATES.items()):
        if i % 2 != partition:
            continue
        for k in range(5):
            folder = OUT/f"results/adaptive/{name}/fold{k}"
            if (folder/"summary.json").exists():
                continue
            tr, va = s["fold"] != k, s["fold"] == k
            assert_split(s, tr, va)
            assert time.monotonic()-start < 3600
            try:
                t = time.monotonic()
                gx, ax = arrays[c["gesture_feature"]], arrays[c["angle_feature"]]
                gs, az = StandardScaler().fit(gx[tr]), StandardScaler().fit(ax[tr])
                g = estimator(base_config(c["gesture_model"], c)).fit(gs.transform(gx[tr]), y[tr]//2)
                a = estimator(base_config(c["angle_model"], c)).fit(az.transform(ax[tr]), y[tr]%2)
                seconds = time.monotonic()-t
                gp = g.predict(gs.transform(gx[va])); ap = a.predict(az.transform(ax[va]))
                pred = 2*gp+ap
                gprob = probabilities(g, gs.transform(gx[va]), c["gesture_model"])
                aprob = probabilities(a, az.transform(ax[va]), c["angle_model"])
                prob = (gprob[:, :, None]*aprob[:, None, :]).reshape(-1, 8)
                # For SVM the explicit hard two-task prediction preserves estimator decisions.
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1)
                train_pred = 2*g.predict(gs.transform(gx[tr]))+a.predict(az.transform(ax[tr]))
                row = {**score(y[va], pred), "name": name, "config": c, "fit_seconds": seconds,
                       "train_accuracy": float(np.mean(train_pred == y[tr])), "timestamp": stamp(),
                       "gesture_columns": gx.shape[1], "angle_columns": ax.shape[1],
                       "svm_margin_surrogate": c["gesture_model"] == "svm" or c["angle_model"] == "svm",
                       "posthoc_probability_calibration": False}
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/"predictions.npz", indices=np.flatnonzero(va), true=y[va],
                                    pred=pred, probabilities=prob, source_indices=s["source_indices"][va])
                write(folder/"summary.json", row)
                print(name, k, f"acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} angle={row['angle_accuracy']:.4f} seconds={seconds:.2f}", flush=True)
            except Exception:
                write(folder/"failure.json", {"timestamp": stamp(), "traceback": traceback.format_exc()})
                raise


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(int(sys.argv[1]))
