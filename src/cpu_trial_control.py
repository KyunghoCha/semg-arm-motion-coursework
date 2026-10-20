"""Matched one-second-context SVM control; CV and holdout are separate commands."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import sklearn
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
OPTIONS = {"mean_c01": ("mean", .1), "mean_c1": ("mean", 1.),
           "mean_c10": ("mean", 10.), "moments_c1": ("moments", 1.),
           "moments_c10": ("moments", 10.)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def features(kind):
    x = np.load(ROOT/"prior/corrected_v3/preprocessing/peak_energy1s70.npz")["X"].copy()
    x[:, :15] = np.log10(np.maximum(x[:, :15], 1e-12))
    x = x.reshape(-1, 9, 70)
    return x.mean(1) if kind == "mean" else np.concatenate([x.mean(1), x.std(1)], axis=1)


def score(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, p, labels=range(8)).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cv", "final"])
    args = parser.parse_args()
    out = ROOT/"results/cpu_trial_control"
    config = {"options": OPTIONS, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    write(out/"config.json", config)
    for protocol in ["recording", "trial"]:
        split = np.load(ROOT/f"data/{protocol}_split.npz")
        y, dev, test, fold = (split[k][::9] for k in ["y", "development", "test", "fold"])
        if args.stage == "cv":
            all_results = {}
            for name, (kind, c) in OPTIONS.items():
                x = features(kind)
                scores = []
                predictions = np.full(len(y), -1)
                for k in range(5):
                    tr, va = dev & (fold != k), dev & (fold == k)
                    model = make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", cache_size=256))
                    start = time.monotonic()
                    model.fit(x[tr], y[tr])
                    predictions[va] = model.predict(x[va])
                    result = score(y[va], predictions[va])
                    result.update(seconds=time.monotonic()-start, fold=k)
                    scores.append(result)
                summary = {"model": name, "protocol": protocol, "folds": scores,
                           "cv_accuracy": float(np.mean([s["accuracy"] for s in scores])),
                           "cv_macro_f1": float(np.mean([s["macro_f1"] for s in scores]))}
                all_results[name] = summary
                write(out/protocol/name/"cv_summary.json", summary)
                np.savez_compressed(out/protocol/name/"cv_predictions.npz",
                                    indices=np.flatnonzero(dev), true=y[dev], pred=predictions[dev],
                                    fold=fold[dev])
                print(protocol, name, summary["cv_accuracy"], summary["cv_macro_f1"], flush=True)
            winner = sorted(OPTIONS, key=lambda n: (-all_results[n]["cv_macro_f1"], n))[0]
            write(out/protocol/"SELECTION_LOCK.json",
                  {"timestamp": datetime.now(timezone.utc).isoformat(), "selected": winner,
                   "all_candidates": all_results})
        else:
            for p in ["recording", "trial"]:
                assert (ROOT/f"results/{p}/FINAL_SELECTION_LOCK.json").exists()
                assert (out/p/"SELECTION_LOCK.json").exists()
            selection = json.loads((out/protocol/"SELECTION_LOCK.json").read_text())
            name = selection["selected"]
            target = out/protocol/name/"final"
            if (target/"summary.json").exists():
                continue
            kind, c = OPTIONS[name]
            x = features(kind)
            model = make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", cache_size=256))
            start = time.monotonic()
            model.fit(x[dev], y[dev])
            pred = model.predict(x[test])
            result = score(y[test], pred)
            result.update(model=name, protocol=protocol, seconds=time.monotonic()-start,
                          cv_macro_f1=selection["all_candidates"][name]["cv_macro_f1"])
            write(target/"summary.json", result)
            joblib.dump(model, target/"model.joblib")
            np.savez_compressed(target/"predictions.npz", true=y[test], pred=pred,
                                trial=split["trial"][::9][test],
                                recording=split["recording"][::9][test], indices=np.flatnonzero(test))
            print("FINAL", protocol, name, result, flush=True)


if __name__ == "__main__":
    main()
