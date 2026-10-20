"""Resumable grouped screening and nested CV, never accesses A/B test inputs."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, time, shutil, traceback, platform
import numpy as np
import sklearn
from models import Model, configurations

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2]
OUT = TASK/"step2_ml_v3"
DATA = OUT/"data"
CONFIG = configurations()


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def score(y, p):
    cm = np.bincount(y*8+p, minlength=64).reshape(8, 8)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(8), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(8), where=cm.sum(1) > 0)
    f1 = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(8), where=cm.sum(0)+cm.sum(1) > 0)
    return {"accuracy": float(np.mean(y == p)), "macro_f1": float(f1.mean()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "gesture_accuracy": float(np.mean(y//2 == p//2)), "angle_accuracy": float(np.mean(y%2 == p%2)),
            "confusion_matrix": cm.tolist(), "n": len(y)}


def freeze():
    frozen = {"config": CONFIG, "sources": {n: sha(HERE/n) for n in ["train.py", "models.py", "prepare.py", "PROTOCOL.md"]},
              "data_manifest_sha256": sha(DATA/"manifest.json"),
              "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}}
    p = OUT/"results/CONFIG_FROZEN.json"
    if p.exists():
        assert json.loads(p.read_text())["frozen"] == frozen
    else:
        write(p, {"timestamp": stamp(), "frozen": frozen})
    return frozen


def assert_split(s, tr, va):
    assert not np.any(tr & va)
    assert set(s["recording"][tr]).isdisjoint(s["recording"][va])
    assert set(s["trial"][tr]).isdisjoint(s["trial"][va])
    assert len(np.unique(s["y"][tr])) == 8


def cases(stage, name, s):
    if stage == "screen":
        for k in range(5):
            yield OUT/f"results/screen/{name}/fold{k}", s["fold"] != k, s["fold"] == k
    elif stage == "inner":
        if not CONFIG[name]["primary"]:
            return
        for outer in range(5):
            pool = s["fold"] != outer
            for k in range(3):
                yield OUT/f"results/inner/outer{outer}/{name}/fold{k}", pool & (s[f"inner{outer}"] != k), pool & (s[f"inner{outer}"] == k)
    elif stage == "outer":
        lock = json.loads((OUT/"results/SELECTION_LOCK.json").read_text())
        for k, row in enumerate(lock["outer"]):
            if name not in set([row["single"], "legacy_svm1", *row["ensemble"]]):
                continue
            yield OUT/f"results/outer/outer{k}/{name}", s["fold"] != k, s["fold"] == k
    else:
        raise ValueError(stage)


def fit(name, cfg, s, x, folder, tr, va):
    assert_split(s, tr, va)
    assert shutil.disk_usage(OUT).free > 5*1024**3
    t = time.monotonic()
    model = Model(cfg).fit(x[tr], s["y"][tr], s["recording"][tr])
    fit_seconds = time.monotonic()-t
    pred = model.predict(x[va])
    prob = model.predict_proba(x[va])
    assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1)
    row = {**score(s["y"][va], pred), "name": name, "config": cfg, "timestamp": stamp(),
           "fit_seconds": fit_seconds, "train_accuracy": float(np.mean(model.predict(x[tr]) == s["y"][tr])),
           "selected_features": model.columns.tolist(), "transformed_features": model.transform(x[va][:1]).shape[1],
           "train_recordings": len(np.unique(s["recording"][tr])), "validation_recordings": len(np.unique(s["recording"][va]))}
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/"predictions.npz", indices=np.flatnonzero(va), source_indices=s["source_indices"][va],
                        true=s["y"][va], pred=pred, probabilities=prob)
    if "/outer/" in str(folder):
        import joblib
        joblib.dump(model, folder/"model.joblib")
    write(folder/"summary.json", row)
    return row


def run(stage, partition):
    freeze()
    s = dict(np.load(DATA/"splits.npz")); start = time.monotonic()
    arrays = {}
    for i, (name, cfg) in enumerate(CONFIG.items()):
        if i % 2 != partition:
            continue
        if cfg["feature"] not in arrays:
            arrays[cfg["feature"]] = np.load(DATA/f"{cfg['feature']}.npy")
        x = arrays[cfg["feature"]]
        for folder, tr, va in cases(stage, name, s):
            assert time.monotonic()-start < 3600, "per-stage60minute resource bound"
            if (folder/"summary.json").exists():
                continue
            try:
                row = fit(name, cfg, s, x, folder, tr, va)
            except Exception:
                write(folder/"failure.json", {"timestamp": stamp(), "name": name, "traceback": traceback.format_exc()})
                raise
            print(stage, str(folder.relative_to(OUT)), f"acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} seconds={row['fit_seconds']:.2f}", flush=True)


def summarize(folder):
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("fold*/summary.json"))]
    assert len(rows) in (3, 5), (folder, len(rows))
    return {"accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "accuracy_sd": float(np.std([r["accuracy"] for r in rows], ddof=1)),
            "train_accuracy": float(np.mean([r["train_accuracy"] for r in rows])),
            "fit_seconds": float(np.sum([r["fit_seconds"] for r in rows])), "folds": rows}


def select():
    freeze()
    p = OUT/"results/SELECTION_LOCK.json"
    assert not p.exists()
    primary = [n for n, c in CONFIG.items() if c["primary"]]
    outer = []
    for k in range(5):
        results = {n: summarize(OUT/f"results/inner/outer{k}/{n}") for n in primary}
        ranking = sorted(primary, key=lambda n: (-results[n]["macro_f1"], n))
        probability_ranking = [n for n in ranking if CONFIG[n]["model"] not in ("svm", "linear")]
        outer.append({"single": ranking[0], "ensemble": probability_ranking[:2], "candidates": results})
    all_results = {n: summarize(OUT/f"results/screen/{n}") for n in CONFIG}
    best = sorted(CONFIG, key=lambda n: (-all_results[n]["macro_f1"], n))[0]
    write(p, {"timestamp": stamp(), "best_development": best, "development": all_results,
              "outer": outer, "config_sha256": sha(OUT/"results/CONFIG_FROZEN.json"),
              "A_B_test_evaluation": False})
    print("LOCKED", best, [(r["single"], r["ensemble"]) for r in outer], flush=True)


def self_check():
    freeze()
    s = dict(np.load(DATA/"splits.npz"))
    for cfg in CONFIG.values():
        assert cfg["feature"] in json.loads((DATA/"manifest.json").read_text())["features"]
    for k in range(5):
        assert_split(s, s["fold"] != k, s["fold"] == k)
        for j in range(3):
            pool = s["fold"] != k
            assert_split(s, pool & (s[f"inner{k}"] != j), pool & (s[f"inner{k}"] == j))
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(8), 20)
    x = rng.normal(size=(160, 80))
    x[:, :8] += np.eye(8)[y]*12
    checks = {}
    for name in ["legacy_svm1", "fusion_et3", "fusion_lda", "fusion_hgb31",
                 "fusion_f32_svm1", "fusion_pca_svm1", "fusion_et3_independent", "fusion_et3_hierarchy"]:
        model = Model(CONFIG[name]).fit(x, y, np.repeat(np.arange(40), 4))
        prob = model.predict_proba(x)
        acc = float(np.mean(model.predict(x) == y))
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1) and acc > .95, (name, acc)
        assert np.array_equal(model.scaler.mean_, x.mean(0)) if hasattr(model.scaler, "mean_") else True
        checks[name] = {"toy_fit_accuracy": acc, "finite_normalized_probabilities": True}
    write(OUT/"reports/self_check.json", checks)
    print("SELF CHECK", len(CONFIG), "candidates;", len(checks), "model paths", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    elif sys.argv[1] == "self-check":
        self_check()
    else:
        run(sys.argv[1], int(sys.argv[2]))
