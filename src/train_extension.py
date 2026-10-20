"""Validation-only bounded extension of the first CNN experiment."""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import torch
import train_gpu as base

ROOT = base.ROOT
EXTRA = ["spectral_relative200", "spectral_aug200",
         "spectral_relative1000", "spectral_aug1000"]
OriginalCNN = base.SignalCNN


class AugmentedCNN(OriginalCNN):
    def forward(self, x):
        if self.training and "_aug" in self.candidate:
            common = torch.randn(len(x), 1, 1, 1, device=x.device)*.30
            channel = torch.randn(len(x), 5, 1, 1, device=x.device)*.15
            x = (x.reshape(len(x), 5, 33, -1) + 2*(common+channel)/math.log(10)).reshape_as(x)
        return super().forward(x)


# The shared fit/restore code resolves this constructor; no base source modification.
base.SignalCNN = AugmentedCNN


def load_input(candidate):
    length = 200 if candidate.endswith("200") else 1000
    raw = np.load(ROOT/f"data/raw{length}.npy")
    x = base.spectral(raw) if candidate.startswith("spectral") else raw
    if "_relative" in candidate:
        x = x-x.mean(axis=(1, 2), keepdims=True)
    return x


def check_extension():
    base.set_seed(17)
    x = torch.randn(32, 165, 9, device="cuda")
    model = AugmentedCNN("spectral_aug200", np.zeros((1,165,1)),
                         np.ones((1,165,1))).cuda()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x)), "augmentation active during inference"
    model.train()
    assert not torch.equal(model(x), model(x))
    raw = np.random.default_rng(5).normal(size=(8, 5, 1000))
    a, b = base.spectral(raw), base.spectral(raw*3)
    a -= a.mean((1,2), keepdims=True)
    b -= b.mean((1,2), keepdims=True)
    assert np.allclose(a, b, atol=2e-5), "relative input is not invariant to common gain"
    base.save_json(ROOT/"results/extension_self_check.json",
                   {"inference_augmentation_disabled": True,
                    "training_augmentation_enabled": True,
                    "relative_common_gain_invariance": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cv", "final"])
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    out = ROOT/"results"
    original = json.loads((out/"CONFIG_FROZEN.json").read_text())
    assert base.digest(ROOT/"train_gpu.py") == original["code_sha256"]
    frozen = {"config": base.CONFIG, "extra_candidates": EXTRA,
              "extension_code_sha256": base.digest(__file__),
              "base_code_sha256": base.digest(ROOT/"train_gpu.py"),
              "extension_protocol_sha256": base.digest(ROOT/"PROTOCOL_EXTENSION.md")}
    freeze = out/"EXTENSION_FROZEN.json"
    if freeze.exists():
        assert json.loads(freeze.read_text()) == frozen
    else:
        base.save_json(freeze, frozen)
    start = time.monotonic()
    if args.stage == "cv":
        check_extension()
        for protocol in base.CONFIG["protocols"]:
            split = dict(np.load(ROOT/f"data/{protocol}_split.npz"))
            for candidate in EXTRA:
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
                        result, model = base.fit(candidate, x_np, x, y, dev & (fold != k),
                                                 dev & (fold == k), folder, 42+k, start)
                        del model
                    folds.append(result)
                    print("EXTENSION FOLD", protocol, candidate, k, result["accuracy"],
                          result["macro_f1"], flush=True)
                summary = {"candidate": candidate, "protocol": protocol, "folds": folds,
                           "mean_accuracy": float(np.mean([f["accuracy"] for f in folds])),
                           "sd_accuracy": float(np.std([f["accuracy"] for f in folds], ddof=1)),
                           "mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
                           "final_epochs": max(1, int(math.floor(np.median(
                               [f["best_epoch"] for f in folds])+.5)))}
                base.save_json(target/"cv_summary.json", summary)
                del x, x_np
                torch.cuda.empty_cache()
        for protocol in base.CONFIG["protocols"]:
            all_results = {}
            for candidate in base.CONFIG["candidates"]+EXTRA:
                all_results[candidate] = json.loads(
                    (out/protocol/candidate/"cv_summary.json").read_text())
            selected = {}
            for unit in [200, 1000]:
                names = [c for c in all_results if c.endswith(str(unit))]
                name = sorted(names, key=lambda c: (-all_results[c]["mean_macro_f1"], c))[0]
                selected[str(unit)] = {"candidate": name, "epochs": all_results[name]["final_epochs"],
                                      "cv_mean_macro_f1": all_results[name]["mean_macro_f1"]}
            base.save_json(out/protocol/"FINAL_SELECTION_LOCK.json",
                           {"timestamp": base.stamp(), "selected": selected,
                            "all_candidates": all_results})
        return
    for protocol in base.CONFIG["protocols"]:
        assert (out/protocol/"FINAL_SELECTION_LOCK.json").exists()
    for protocol in base.CONFIG["protocols"]:
        split = dict(np.load(ROOT/f"data/{protocol}_split.npz"))
        locked = json.loads((out/protocol/"FINAL_SELECTION_LOCK.json").read_text())
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
            for seed in base.CONFIG["final_seeds"]:
                folder = final_dir/f"seed{seed}"
                if (folder/"summary.json").exists():
                    training = json.loads((folder/"summary.json").read_text())
                    model = base.restore(folder/"last.pt")
                else:
                    training, model = base.fit(candidate, x_np, x, y, dev, None, folder,
                                               seed, start, epochs=choice["epochs"])
                prob = base.predict(model, x, np.flatnonzero(test))
                probs.append(prob)
                seed_result = base.metrics(y[test], prob.argmax(1))
                if unit == "200":
                    ty, tp = base.trial_vote(y[test], prob)
                    seed_result["trial_vote"] = base.metrics(ty, tp)
                seed_results.append({"seed": seed, "test": seed_result, "training": training})
                del model
            average = np.mean(probs, axis=0)
            result = base.metrics(y[test], average.argmax(1))
            if unit == "200":
                ty, tp = base.trial_vote(y[test], average)
                result["trial_vote"] = base.metrics(ty, tp)
            result.update(candidate=candidate, protocol=protocol, input_ms=int(unit),
                          ensemble="equal softmax mean of the two predeclared seeds",
                          seeds=seed_results, selection=choice,
                          selection_sha256=base.digest(out/protocol/"FINAL_SELECTION_LOCK.json"),
                          test_is_reused=True)
            np.savez_compressed(final_dir/"predictions.npz", true=y[test],
                                probabilities=average, seed_probabilities=np.stack(probs),
                                indices=np.flatnonzero(test),
                                recording=split["recording"][::step][test],
                                trial=split["trial"][::step][test],
                                window_start=split["window_start"][::step][test])
            base.save_json(final_dir/"summary.json", result)
            print("FINAL", protocol, candidate, result["accuracy"], result["macro_f1"], flush=True)
            del x, x_np
            torch.cuda.empty_cache()
    print("STAGE COMPLETE", args.stage, base.stamp(), flush=True)


if __name__ == "__main__":
    main()
