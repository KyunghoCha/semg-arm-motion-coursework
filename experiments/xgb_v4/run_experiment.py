"""Frozen, resumable CUDA XGBoost comparison using only A development features."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import json, sys, time, shutil, traceback, subprocess, platform
import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "ml_v3"))
from train import score, stamp, sha, write, assert_split, summarize

OUT = TASK / "step2_xgb_v4"
DATA = TASK / "step2_ml_v3/data"
CONFIG = {f"{feature}_{structure}_d{depth}":
          {"feature": feature, "structure": structure, "depth": depth}
          for feature in ("legacy", "robust", "fusion", "relative")
          for structure in ("joint", "independent") for depth in (2, 4, 6)}
PARAMS = dict(tree_method="hist", device="cuda:0", eta=.05,
              min_child_weight=5, reg_lambda=5, subsample=.8,
              colsample_bytree=.8, max_bin=128, nthread=2,
              objective="multi:softprob", eval_metric="mlogloss", seed=20261024)


def resource():
    line = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    temp, mem, util = map(int, line.split(","))
    assert temp < 85, ("GPU temperature", temp)
    assert shutil.disk_usage(OUT).free > 5 * 1024**3
    return dict(timestamp=stamp(), temperature_c=temp, device_memory_mib=mem, utilization=util)


def freeze():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["rows"] == 1530 and manifest["records"] == 51 and manifest["original_test_rows"] == 0
    for n in ["splits.npz", *[f"{c}.npy" for c in ("legacy", "robust", "fusion", "relative")]]:
        assert sha(DATA / n) == manifest["hashes"][n], n
    frozen = dict(config=CONFIG, params=PARAMS, rounds=200,
                  sources={n: sha(HERE / n) for n in ("run_experiment.py", "PROTOCOL.md")},
                  reused_helpers_sha256=sha(HERE.parent / "ml_v3/train.py"),
                  manifest_sha256=sha(DATA / "manifest.json"),
                  diagnostic_splits_sha256=sha(TASK / "step2_ml_v3/results/diagnostic_splits.npz"),
                  environment=dict(python=platform.python_version(), xgboost=xgb.__version__, numpy=np.__version__))
    p = OUT / "results/CONFIG_FROZEN.json"
    if p.exists():
        assert json.loads(p.read_text())["frozen"] == frozen, "frozen settings changed"
    else:
        write(p, dict(timestamp=stamp(), frozen=frozen, A_B_test_evaluation=False))
    return frozen


class Model:
    def __init__(self, config):
        self.config = config
        self.heads = []
        self.curves = []

    def fit(self, x, y, vx=None, vy=None, rounds=200):
        labels = [(y, vy, 8)] if self.config["structure"] == "joint" else [
            (y // 2, None if vy is None else vy // 2, 4),
            (y % 2, None if vy is None else vy % 2, 2)]
        for i, (target, vtarget, classes) in enumerate(labels):
            counts = np.bincount(target, minlength=classes)
            assert (counts > 0).all()
            weights = len(y) / (classes * counts[target])
            train = xgb.DMatrix(x, label=target, weight=weights, nthread=2)
            evals = [(train, "train")]
            if vx is not None:
                evals.append((xgb.DMatrix(vx, label=vtarget, nthread=2), "validation"))
            history = {}
            params = {**PARAMS, "max_depth": self.config["depth"], "num_class": classes,
                      "seed": PARAMS["seed"] + i}
            model = xgb.train(params, train, num_boost_round=rounds, evals=evals,
                              evals_result=history, verbose_eval=False)
            assert json.loads(model.save_config())["learner"]["generic_param"]["device"] == "cuda:0"
            self.heads.append(model)
            self.curves.append(history)
        return self

    def predict_proba(self, x):
        d = xgb.DMatrix(x, nthread=2)
        parts = [m.predict(d) for m in self.heads]
        return parts[0] if len(parts) == 1 else (parts[0][:, :, None] * parts[1][:, None, :]).reshape(-1, 8)

    def save(self, folder):
        for i, head in enumerate(self.heads):
            head.save_model(folder / f"head{i}.ubj")


def fit(name, cfg, s, folder, tr, va, save=False):
    assert_split(s, tr, va)
    x = np.load(DATA / f"{cfg['feature']}.npy").astype(np.float32)
    assert np.isfinite(x).all()
    before = resource()
    t = time.monotonic()
    model = Model(cfg).fit(x[tr], s["y"][tr], x[va], s["y"][va])
    elapsed = time.monotonic() - t
    prob = model.predict_proba(x[va])
    assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=2e-6)
    pred = prob.argmax(1)
    train_pred = model.predict_proba(x[tr]).argmax(1)
    row = {**score(s["y"][va], pred), "name": name, "config": cfg, "timestamp": stamp(),
           "fit_seconds": elapsed, "train_accuracy": float(np.mean(train_pred == s["y"][tr])),
           "train_recordings": len(np.unique(s["recording"][tr])),
           "validation_recordings": len(np.unique(s["recording"][va])),
           "resources": [before, resource()], "device": "cuda:0", "classes_present": np.unique(s["y"][va]).tolist()}
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", indices=np.flatnonzero(va),
                        true=s["y"][va], pred=pred, probabilities=prob)
    write(folder / "curves.json", model.curves)
    if save:
        model.save(folder)
    write(folder / "summary.json", row)
    return row


def cases(stage, s):
    if stage == "screen":
        for name, cfg in CONFIG.items():
            for k in range(5):
                yield name, cfg, f"screen/{name}/fold{k}", s["fold"] != k, s["fold"] == k
    elif stage == "inner":
        for outer in range(5):
            pool = s["fold"] != outer
            for name, cfg in CONFIG.items():
                for k in range(3):
                    yield name, cfg, f"inner/outer{outer}/{name}/fold{k}", pool & (s[f"inner{outer}"] != k), pool & (s[f"inner{outer}"] == k)
    elif stage == "outer":
        lock = json.loads((OUT / "results/SELECTION_LOCK.json").read_text())
        for k, row in enumerate(lock["outer"]):
            name = row["name"]
            yield name, CONFIG[name], f"outer/fold{k}", s["fold"] != k, s["fold"] == k
    elif stage == "diagnostics":
        lock = json.loads((OUT / "results/SELECTION_LOCK.json").read_text())
        name = lock["best_development"]
        ds = dict(np.load(TASK / "step2_ml_v3/results/diagnostic_splits.npz"))
        for seed, f in ds.items():
            for k in range(5):
                yield name, CONFIG[name], f"diagnostics/record_seed{seed}/fold{k}", f != k, f == k
        for person in np.unique(s["subject"]):
            yield name, CONFIG[name], f"diagnostics/subject/{person}", s["subject"] != person, s["subject"] == person
    else:
        raise ValueError(stage)


def run(stage):
    freeze()
    s = dict(np.load(DATA / "splits.npz"))
    start = time.monotonic()
    for name, cfg, label, tr, va in cases(stage, s):
        folder = OUT / "results" / label
        if (folder / "summary.json").exists():
            continue
        try:
            assert time.monotonic() - start < 3600
            if "/subject/" in label:
                assert set(s["subject"][tr]).isdisjoint(s["subject"][va])
            row = fit(name, cfg, s, folder, tr, va, save=stage == "outer")
        except Exception:
            write(folder / "failure.json", dict(timestamp=stamp(), traceback=traceback.format_exc()))
            raise
        print(stamp(), label, f"acc={row['accuracy']:.5f} f1={row['macro_f1']:.5f} seconds={row['fit_seconds']:.2f}", flush=True)


def select():
    freeze()
    p = OUT / "results/SELECTION_LOCK.json"
    assert not p.exists()
    development = {n: summarize(OUT / f"results/screen/{n}") for n in CONFIG}
    rank = lambda rows: sorted(rows, key=lambda n: (-rows[n]["macro_f1"], n))
    outer = []
    for k in range(5):
        rows = {n: summarize(OUT / f"results/inner/outer{k}/{n}") for n in CONFIG}
        outer.append(dict(name=rank(rows)[0], candidates=rows))
    write(p, dict(timestamp=stamp(), best_development=rank(development)[0], development=development,
                  outer=outer, A_B_test_evaluation=False,
                  config_sha256=sha(OUT / "results/CONFIG_FROZEN.json")))
    print("LOCKED", rank(development)[0], [o["name"] for o in outer], flush=True)


def representative():
    freeze()
    folder = OUT / "results/representative"
    assert not folder.exists()
    name = json.loads((OUT / "results/SELECTION_LOCK.json").read_text())["best_development"]
    s = dict(np.load(DATA / "splits.npz"))
    x = np.load(DATA / f"{CONFIG[name]['feature']}.npy").astype(np.float32)
    resource(); t = time.monotonic()
    m = Model(CONFIG[name]).fit(x, s["y"])
    elapsed = time.monotonic() - t
    folder.mkdir(parents=True)
    m.save(folder)
    write(folder / "summary.json", dict(name=name, config=CONFIG[name], timestamp=stamp(),
          fit_seconds=elapsed, n=len(x), evaluation="none; all development training only", resources=[resource()]))


def self_check():
    freeze()
    s = dict(np.load(DATA / "splits.npz"))
    for stage in ("screen", "inner"):
        for _, _, _, tr, va in cases(stage, s):
            assert_split(s, tr, va)
    rng = np.random.default_rng(42)
    y = np.repeat(np.arange(8), 24)
    x = rng.normal(size=(len(y), 16)).astype(np.float32)
    x[:, :8] += np.eye(8, dtype=np.float32)[y] * 12
    checks = {}
    folder = OUT / "self_check"
    folder.mkdir(exist_ok=True)
    for structure in ("joint", "independent"):
        m = Model(dict(structure=structure, depth=3)).fit(x, y, rounds=80)
        prob = m.predict_proba(x)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=2e-6)
        acc = float(np.mean(prob.argmax(1) == y))
        assert acc > .95
        m.save(folder)
        loaded = Model(m.config)
        loaded.heads = [xgb.Booster(model_file=folder / f"head{i}.ubj") for i in range(len(m.heads))]
        assert np.array_equal(prob, loaded.predict_proba(x))
        checks[structure] = dict(toy_accuracy=acc, replay_exact=True)
    write(OUT / "reports/self_check.json", dict(checks=checks, splits_checked=480, resource=resource()))
    print("SELF CHECK PASSED", checks, flush=True)


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "self-check": self_check()
    elif action == "select": select()
    elif action == "representative": representative()
    else: run(action)
