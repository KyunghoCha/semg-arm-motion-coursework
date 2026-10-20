"""Bounded local-GPU CV, locked selection, then one frozen holdout evaluation.

No CPU fallback; all trainable models must run on the user's CUDA device.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "candidates": ["raw200", "spectral200", "raw1000", "spectral1000"],
    "protocols": ["recording", "trial"],
    "batch_size": 128, "lr": .001, "weight_decay": .001,
    "dropout": .25, "label_smoothing": .05, "grad_clip": 5.,
    "max_epochs": 40, "min_epochs": 8, "patience": 8,
    "cv_seed": 42, "final_seeds": [42, 2026], "max_wall_seconds": 5400,
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def spectral(x):
    """STFT within each sample only; no padding, fitted quantities or outside context."""
    frames = np.lib.stride_tricks.sliding_window_view(x, 64, axis=-1)[..., ::16, :]
    power = np.abs(np.fft.rfft(frames * np.hanning(64), axis=-1))**2
    log_power = np.log10(np.maximum(power, 1e-12))
    return log_power.transpose(0, 1, 3, 2).reshape(len(x), 165, -1).astype("float32")


def metrics(y, pred):
    cm = np.bincount(y * 8 + pred, minlength=64).reshape(8, 8)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*tp, denom, out=np.zeros(8), where=denom != 0)
    return {"accuracy": float(tp.sum()/cm.sum()), "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(), "n": int(len(y)),
            "per_class_f1": f1.tolist()}


def trial_vote(y, prob):
    assert len(y) % 9 == 0 and np.all(y.reshape(-1, 9) == y.reshape(-1, 9)[:, :1])
    p = prob.argmax(1).reshape(-1, 9)
    vote = np.array([np.bincount(row, minlength=8).argmax() for row in p])
    return y[::9], vote


class SignalCNN(nn.Module):
    def __init__(self, candidate, mean, scale):
        super().__init__()
        self.candidate = candidate
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.raw = candidate.startswith("raw")
        if self.raw:
            self.body = nn.Sequential(
                nn.Conv1d(5, 40, 15, stride=2, padding=7, groups=5),
                nn.BatchNorm1d(40), nn.LeakyReLU(.1),
                nn.Conv1d(40, 64, 9, stride=2, padding=4),
                nn.BatchNorm1d(64), nn.LeakyReLU(.1),
                nn.Conv1d(64, 64, 5, padding=2),
                nn.BatchNorm1d(64), nn.LeakyReLU(.1),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv1d(165, 64, 1),
                nn.BatchNorm1d(64), nn.LeakyReLU(.1),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.BatchNorm1d(64), nn.LeakyReLU(.1),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.BatchNorm1d(64), nn.LeakyReLU(.1),
            )
        self.head = nn.Sequential(nn.Linear(133 if self.raw else 128, 64),
                                  nn.LeakyReLU(.1), nn.Dropout(CONFIG["dropout"]),
                                  nn.Linear(64, 8))

    def forward(self, x):
        normalized = (x-self.mean)/self.scale
        h = self.body(normalized)
        summary = [h.mean(dim=-1), h.amax(dim=-1)]
        if self.raw:
            summary.append(torch.log10(torch.mean(normalized**2, dim=-1).clamp_min(1e-12).sqrt()))
        return self.head(torch.cat(summary, dim=1))


def normalization(x, train):
    # Copy only training rows; validation and test never contribute statistics.
    tx = x[train].astype("float64")
    mean = tx.mean(axis=(0, 2), keepdims=True).astype("float32")
    scale = tx.std(axis=(0, 2), keepdims=True).clip(1e-8).astype("float32")
    return mean, scale


@torch.no_grad()
def predict(model, x, idx):
    model.eval()
    result = []
    for batch in torch.as_tensor(idx, device="cuda").split(CONFIG["batch_size"]):
        result.append(model(x[batch]).softmax(1).cpu().numpy())
    return np.concatenate(result)


def resource_check(start):
    if time.monotonic()-start > CONFIG["max_wall_seconds"]:
        raise RuntimeError("Predeclared 90 minute experiment budget exceeded.")
    if shutil.disk_usage(ROOT).free < 5*1024**3:
        raise RuntimeError("Free disk fell below 5 GiB.")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    temperature, used, utilization = map(int, result.stdout.strip().splitlines()[0].split(","))
    assert temperature < 85, f"GPU temperature {temperature} C"
    assert torch.cuda.max_memory_allocated() < 8*1024**3
    return {"temperature_c": temperature, "device_memory_used_mib": used,
            "device_utilization_pct": utilization,
            "process_peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2}


def save_checkpoint(path, model, candidate, epoch, seed):
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "candidate": candidate,
                "epoch": epoch, "seed": seed, "config": CONFIG}, tmp)
    tmp.replace(path)


def restore(path):
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    state = checkpoint["state_dict"]
    model = SignalCNN(checkpoint["candidate"], state["mean"], state["scale"]).cuda()
    model.load_state_dict(state)
    return model


def fit(candidate, x_np, x, y, tr, va, out, seed, budget_start, epochs=None):
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    mean, scale = normalization(x_np, tr)
    model = SignalCNN(candidate, mean, scale).cuda()
    params = sum(p.numel() for p in model.parameters())
    assert params < 250_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    y_gpu = torch.as_tensor(y, device="cuda")
    train = torch.as_tensor(np.flatnonzero(tr), device="cuda")
    best, best_epoch, stale = -1., 0, 0
    t0 = time.monotonic()
    with (out/"epochs.jsonl").open("w") as log:
        for epoch in range(1, (epochs or CONFIG["max_epochs"])+1):
            model.train()
            permutation = train[torch.randperm(len(train), device="cuda")]
            total_loss = total_correct = seen = 0
            for batch in permutation.split(CONFIG["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[batch])
                loss = criterion(logits, y_gpu[batch])
                assert torch.isfinite(loss), "nonfinite training loss"
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item()*len(batch)
                total_correct += int((logits.argmax(1) == y_gpu[batch]).sum().item())
                seen += len(batch)
            row = {"timestamp": stamp(), "epoch": epoch, "train_loss": total_loss/seen,
                   "train_accuracy": total_correct/seen, "seconds": time.monotonic()-t0}
            if va is not None:
                prob = predict(model, x, np.flatnonzero(va))
                val = metrics(y[va], prob.argmax(1))
                row.update(val_accuracy=val["accuracy"], val_macro_f1=val["macro_f1"])
                score = val["macro_f1"]
                if score > best + 1e-12:
                    best, best_epoch, stale = score, epoch, 0
                    save_checkpoint(out/"best.pt", model, candidate, epoch, seed)
                else:
                    stale += 1
            row.update(resource_check(budget_start))
            log.write(json.dumps(row)+"\n")
            log.flush()
            save_checkpoint(out/"last.pt", model, candidate, epoch, seed)
            if epoch == 1 or epoch % 5 == 0:
                print(str(out.relative_to(ROOT)), json.dumps(row), flush=True)
            if va is not None and epoch >= CONFIG["min_epochs"] and stale >= CONFIG["patience"]:
                break
    if va is not None:
        model = restore(out/"best.pt")
        prob = predict(model, x, np.flatnonzero(va))
        result = metrics(y[va], prob.argmax(1))
        if candidate.endswith("200"):
            ty, tp = trial_vote(y[va], prob)
            result["trial_vote"] = metrics(ty, tp)
        np.savez_compressed(out/"validation.npz", indices=np.flatnonzero(va),
                            true=y[va], probabilities=prob)
    else:
        result = {}
        best_epoch = epochs
    result.update(best_epoch=best_epoch, epochs_completed=epoch, seed=seed, parameters=params,
                  seconds=time.monotonic()-t0, normalization_train_rows=int(tr.sum()),
                  **resource_check(budget_start))
    save_json(out/"summary.json", result)
    return result, model


def self_check():
    assert metrics(np.arange(8), np.arange(8))["macro_f1"] == 1.
    assert metrics(np.arange(8), (np.arange(8)+1) % 8)["accuracy"] == 0.
    y = np.repeat([0, 1], 9)
    predictions = np.array([0]*5+[1]*4+[2]*4+[1]*4+[3])
    assert trial_vote(y, np.eye(8)[predictions])[1].tolist() == [0, 1]
    rng = np.random.default_rng(7)
    out = ROOT/"results/self_check"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for candidate in CONFIG["candidates"]:
        set_seed(7)
        length = 200 if candidate.endswith("200") else 1000
        array = rng.normal(size=(32, 5, length)).astype("float32")
        labels = np.arange(32) % 8
        array[:, 0] += labels[:, None]*1.5
        if candidate.startswith("spectral"):
            array = spectral(array)
        mean, scale = normalization(array, np.ones(32, dtype=bool))
        x, y = torch.from_numpy(array).cuda(), torch.tensor(labels, device="cuda")
        model = SignalCNN(candidate, mean, scale).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        model.train()
        initial = float(nn.functional.cross_entropy(model(x), y).item())
        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            before = model(x)
            final_loss = float(nn.functional.cross_entropy(before, y).item())
            accuracy = float((before.argmax(1) == y).float().mean().item())
        assert before.shape == (32, 8) and torch.isfinite(before).all()
        assert final_loss < initial*.5 and accuracy >= .9, (candidate, initial, final_loss, accuracy)
        save_checkpoint(out/f"{candidate}.pt", model, candidate, 100, 7)
        loaded = restore(out/f"{candidate}.pt").eval()
        with torch.no_grad():
            error = float((loaded(x)-before).abs().max().item())
        assert error == 0
        results[candidate] = {"initial_loss": initial, "final_loss": final_loss,
                              "training_accuracy": accuracy, "reload_max_abs_error": error,
                              "parameters": sum(p.numel() for p in model.parameters())}
    save_json(out/"results.json", results)
    print(json.dumps(results, indent=2), flush=True)


def load_input(candidate):
    length = 200 if candidate.endswith("200") else 1000
    x = np.load(ROOT/f"data/raw{length}.npy")
    return spectral(x) if candidate.startswith("spectral") else x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["self-check", "cv", "final"])
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU unavailable: do not silently use CPU."
    torch.set_num_threads(4)
    out = ROOT/"results"
    out.mkdir(exist_ok=True)
    env = {"time": stamp(), "python": platform.python_version(), "python_executable": os.sys.executable,
           "torch": torch.__version__, "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
           "compute_capability": torch.cuda.get_device_capability(),
           "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory}
    save_json(out/f"environment_{args.stage}.json", env)
    if args.stage == "self-check":
        self_check()
        return
    config_path = out/"CONFIG_FROZEN.json"
    frozen = {"config": CONFIG, "protocol_sha256": digest(ROOT/"PROTOCOL_GPU.md"),
              "code_sha256": digest(__file__), "data_sha256": json.loads((ROOT/"data/sha256.json").read_text())}
    if config_path.exists():
        assert json.loads(config_path.read_text()) == frozen, "Frozen code/config/data differs."
    else:
        save_json(config_path, frozen)
    budget_start = time.monotonic()
    selections = {}
    for protocol in CONFIG["protocols"]:
        split = dict(np.load(ROOT/f"data/{protocol}_split.npz"))
        if args.stage == "cv":
            all_results = {}
            for candidate in CONFIG["candidates"]:
                target = out/protocol/candidate
                target.mkdir(parents=True, exist_ok=True)
                step = 9 if candidate.endswith("1000") else 1
                y = split["y"][::step]
                dev, fold = split["development"][::step], split["fold"][::step]
                x_np = load_input(candidate)
                x = torch.from_numpy(x_np).cuda()
                folds = []
                for k in range(5):
                    folder = target/f"fold{k}"
                    if (folder/"summary.json").exists():
                        result = json.loads((folder/"summary.json").read_text())
                    else:
                        result, model = fit(candidate, x_np, x, y, dev & (fold != k),
                                            dev & (fold == k), folder, CONFIG["cv_seed"]+k, budget_start)
                        del model
                    folds.append(result)
                    print("FOLD COMPLETE", protocol, candidate, k, result["accuracy"],
                          result["macro_f1"], result["seconds"], flush=True)
                summary = {"candidate": candidate, "protocol": protocol, "folds": folds,
                           "mean_accuracy": float(np.mean([f["accuracy"] for f in folds])),
                           "sd_accuracy": float(np.std([f["accuracy"] for f in folds], ddof=1)),
                           "mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
                           "final_epochs": max(1, int(math.floor(np.median(
                               [f["best_epoch"] for f in folds])+.5)))}
                save_json(target/"cv_summary.json", summary)
                all_results[candidate] = summary
                del x, x_np
                torch.cuda.empty_cache()
            selected = {}
            for unit in [200, 1000]:
                names = [c for c in CONFIG["candidates"] if c.endswith(str(unit))]
                name = sorted(names, key=lambda c: (-all_results[c]["mean_macro_f1"], c))[0]
                selected[str(unit)] = {"candidate": name, "epochs": all_results[name]["final_epochs"],
                                       "cv_mean_macro_f1": all_results[name]["mean_macro_f1"]}
            selections[protocol] = {"timestamp": stamp(), "selected": selected,
                                    "all_candidates": all_results}
            save_json(out/protocol/"SELECTION_LOCK.json", selections[protocol])
        else:
            # Both protocols must be selected before any holdout inference.
            for p in CONFIG["protocols"]:
                assert (out/p/"SELECTION_LOCK.json").exists()
            locked = json.loads((out/protocol/"SELECTION_LOCK.json").read_text())
            for unit, choice in locked["selected"].items():
                candidate = choice["candidate"]
                final_dir = out/protocol/candidate/"final"
                if (final_dir/"summary.json").exists():
                    continue
                step = 9 if unit == "1000" else 1
                y = split["y"][::step]
                dev, test = split["development"][::step], split["test"][::step]
                x_np = load_input(candidate)
                x = torch.from_numpy(x_np).cuda()
                probs, seed_results = [], []
                for seed in CONFIG["final_seeds"]:
                    folder = final_dir/f"seed{seed}"
                    if (folder/"summary.json").exists():
                        train_result = json.loads((folder/"summary.json").read_text())
                        model = restore(folder/"last.pt")
                    else:
                        train_result, model = fit(candidate, x_np, x, y, dev, None, folder,
                                                  seed, budget_start, epochs=choice["epochs"])
                    prob = predict(model, x, np.flatnonzero(test))
                    probs.append(prob)
                    seed_result = metrics(y[test], prob.argmax(1))
                    if unit == "200":
                        ty, tp = trial_vote(y[test], prob)
                        seed_result["trial_vote"] = metrics(ty, tp)
                    seed_results.append({"seed": seed, "test": seed_result, "training": train_result})
                    del model
                average = np.mean(probs, axis=0)
                result = metrics(y[test], average.argmax(1))
                if unit == "200":
                    ty, tp = trial_vote(y[test], average)
                    result["trial_vote"] = metrics(ty, tp)
                result.update(candidate=candidate, protocol=protocol, input_ms=int(unit),
                              ensemble="equal softmax probability mean of predeclared seeds",
                              seeds=seed_results, selection=choice,
                              selection_sha256=digest(out/protocol/"SELECTION_LOCK.json"),
                              test_is_reused=True)
                np.savez_compressed(final_dir/"predictions.npz", true=y[test],
                                    probabilities=average, seed_probabilities=np.stack(probs),
                                    indices=np.flatnonzero(test),
                                    recording=split["recording"][::step][test],
                                    trial=split["trial"][::step][test],
                                    window_start=split["window_start"][::step][test])
                save_json(final_dir/"summary.json", result)
                print("FINAL", protocol, candidate, json.dumps(result), flush=True)
                del x, x_np
                torch.cuda.empty_cache()
    print("STAGE COMPLETE", args.stage, stamp(), flush=True)


if __name__ == "__main__":
    main()
