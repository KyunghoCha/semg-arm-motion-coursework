"""Fixed-model resplit/held-person sensitivity; not independent model-selection evaluation."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time, json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import adaptive
from models import Model, estimator
from train import *


def prepare():
    adaptive.frozen()
    p = OUT/"results/DIAGNOSTICS_FROZEN.json"
    assert not p.exists()
    lock = json.loads((OUT/"results/DEVELOPMENT_SELECTION_LOCK.json").read_text())
    original = json.loads((OUT/"results/SELECTION_LOCK.json").read_text())
    five_second = [n for n in lock["rank"] if n in adaptive.CANDIDATES and adaptive.CANDIDATES[n]["context"] == "5s"][0]
    models = sorted(set(["legacy_svm1", original["best_development"], lock["best_1s"], five_second]))
    s = dict(np.load(DATA/"splits.npz"))
    records = np.unique(s["recording"])
    labels = np.array([s["y"][s["recording"] == r][0] for r in records])
    folds = {}
    for seed in [20261022, 20261023]:
        fold = np.full(len(s["y"]), -1, int)
        for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(records, labels)):
            fold[np.isin(s["recording"], records[va])] = k
        folds[str(seed)] = fold
    np.savez(OUT/"results/diagnostic_splits.npz", **folds)
    write(p, {"timestamp": stamp(), "models": models, "subjects": sorted(np.unique(s["subject"]).tolist()),
              "seeds": [20261022, 20261023], "source_sha256": sha(__file__),
              "plan_sha256": sha(HERE/"DIAGNOSTICS_PLAN.md"),
              "split_sha256": sha(OUT/"results/diagnostic_splits.npz"),
              "development_selection_sha256": sha(OUT/"results/DEVELOPMENT_SELECTION_LOCK.json"),
              "interpretation": "fixed-model exploratory split sensitivity; prior model selection used these development data"})
    print("DIAGNOSTICS FROZEN", models, flush=True)


def run(partition):
    adaptive.frozen()
    lock = json.loads((OUT/"results/DIAGNOSTICS_FROZEN.json").read_text())
    assert lock["source_sha256"] == sha(__file__) and lock["plan_sha256"] == sha(HERE/"DIAGNOSTICS_PLAN.md")
    assert lock["split_sha256"] == sha(OUT/"results/diagnostic_splits.npz")
    s = dict(np.load(DATA/"splits.npz")); y = s["y"]
    arrays = {p.stem: np.load(p) for p in DATA.glob("*.npy")}
    arrays.update(dict(np.load(DATA/"adaptive_features.npz")))
    ds = dict(np.load(OUT/"results/diagnostic_splits.npz"))
    cases = [(f"record_seed{seed}/fold{k}", f != k, f == k) for seed, f in ds.items() for k in range(5)]
    cases += [(f"subject/{person}", s["subject"] != person, s["subject"] == person) for person in lock["subjects"]]
    for i, name in enumerate(lock["models"]):
        if i % 2 != partition:
            continue
        for label, tr, va in cases:
            folder = OUT/f"results/diagnostics/{name}/{label}"
            if (folder/"summary.json").exists():
                continue
            assert_split(s, tr, va)
            if label.startswith("subject/"):
                assert set(s["subject"][tr]).isdisjoint(s["subject"][va])
            t = time.monotonic()
            if name in CONFIG:
                c = CONFIG[name]; x = arrays[c["feature"]]
                model = Model(c).fit(x[tr], y[tr], s["recording"][tr])
                seconds = time.monotonic()-t
                pred = model.predict(x[va])
                prob = model.predict_proba(x[va])
            else:
                c = adaptive.CANDIDATES[name]
                gx, ax = arrays[c["gesture_feature"]], arrays[c["angle_feature"]]
                gs, az = StandardScaler().fit(gx[tr]), StandardScaler().fit(ax[tr])
                g = estimator(adaptive.base_config(c["gesture_model"], c)).fit(gs.transform(gx[tr]), y[tr]//2)
                a = estimator(adaptive.base_config(c["angle_model"], c)).fit(az.transform(ax[tr]), y[tr]%2)
                seconds = time.monotonic()-t
                pred = 2*g.predict(gs.transform(gx[va]))+a.predict(az.transform(ax[va]))
                prob = (adaptive.probabilities(g, gs.transform(gx[va]), c["gesture_model"])[:, :, None]*
                        adaptive.probabilities(a, az.transform(ax[va]), c["angle_model"])[:, None, :]).reshape(-1, 8)
            row = {**score(y[va], pred), "name": name, "diagnostic": label, "fit_seconds": seconds,
                   "timestamp": stamp(), "classes_present": np.unique(y[va]).tolist(),
                   "train_records": len(set(s["recording"][tr])), "validation_records": len(set(s["recording"][va]))}
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/"predictions.npz", indices=np.flatnonzero(va), true=y[va], pred=pred, probabilities=prob)
            write(folder/"summary.json", row)
            print(name, label, f"accuracy={row['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(int(sys.argv[1]))
