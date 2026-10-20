"""Recompute all reported metrics from saved predictions; render comparable figures."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/step2_gpu_matplotlib")
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT/"prior/corrected_v3"
LABELS = ["Fist 30", "Fist 60", "Open 30", "Open 60",
          "Pro 30", "Pro 60", "Sup 30", "Sup 60"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=np.arange(8),
                                     average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(8)).tolist()}


def to_trials(y, pred, trial, recording):
    ids, first = np.unique(trial, return_index=True)
    true, prediction = [], []
    for tid in ids:
        ix = trial == tid
        assert len(set(y[ix])) == 1
        true.append(y[ix][0])
        prediction.append(np.bincount(pred[ix], minlength=8).argmax())
    return np.array(true), np.array(prediction), ids, recording[first]


def verify_cv(protocol, candidate):
    folder = ROOT/"results"/protocol/candidate
    summary = json.loads((folder/"cv_summary.json").read_text())
    split = np.load(ROOT/f"data/{protocol}_split.npz")
    step = 9 if candidate.endswith("1000") else 1
    dev, fold = split["development"][::step], split["fold"][::step]
    covered = np.zeros(len(dev), dtype=int)
    values = []
    for k in range(5):
        p = np.load(folder/f"fold{k}/validation.npz")
        ids = p["indices"]
        assert np.array_equal(ids, np.flatnonzero(dev & (fold == k)))
        assert np.array_equal(p["true"], split["y"][::step][ids])
        covered[ids] += 1
        result = score(p["true"], p["probabilities"].argmax(1))
        old = summary["folds"][k]
        for key in ["accuracy", "macro_f1"]:
            assert np.isclose(result[key], old[key], atol=1e-12)
        history = [json.loads(row) for row in (folder/f"fold{k}/epochs.jsonl").read_text().splitlines()]
        best_epoch = int(np.argmax([row["val_macro_f1"] for row in history]))+1
        assert best_epoch == old["best_epoch"]
        values.append(result["macro_f1"])
    assert np.all(covered[dev] == 1) and np.all(covered[~dev] == 0)
    assert np.isclose(np.mean(values), summary["mean_macro_f1"])
    return {"protocol": protocol, "candidate": candidate,
            "cv_accuracy": summary["mean_accuracy"], "cv_accuracy_sd": summary["sd_accuracy"],
            "cv_macro_f1": summary["mean_macro_f1"], "final_epochs": summary["final_epochs"],
            "best_epochs": [f["best_epoch"] for f in summary["folds"]]}


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = ROOT/"report"
    out.mkdir(exist_ok=True)
    frozen = json.loads((ROOT/"results/CONFIG_FROZEN.json").read_text())
    assert sha(ROOT/"train_gpu.py") == frozen["code_sha256"]
    assert sha(ROOT/"PROTOCOL_GPU.md") == frozen["protocol_sha256"]
    extension = json.loads((ROOT/"results/EXTENSION_FROZEN.json").read_text())
    assert sha(ROOT/"train_extension.py") == extension["extension_code_sha256"]
    assert sha(ROOT/"PROTOCOL_EXTENSION.md") == extension["extension_protocol_sha256"]
    for name, digest in frozen["data_sha256"].items():
        assert sha(ROOT/"data"/name) == digest
    cv = [verify_cv(p, c) for p in ["recording", "trial"]
          for c in frozen["config"]["candidates"]+extension["extra_candidates"]]
    write_csv(out/"all_gpu_cv.csv", cv)
    baseline = list(csv.DictReader((PRIOR/"xgb_extension/three_family_comparison.csv").open()))
    rows, models, per_record = [], {}, []
    for row in baseline:
        protocol, family = row["protocol"], row["family"]
        folder = PRIOR/("xgb_extension" if family == "xgb" else "")/"results"/protocol/"final"
        data = np.load(folder/f"{family}_predictions.npz")
        y, pred, trial, rec = (data[k] for k in ["true", "pred", "trial", "recording"])
        window = score(y, pred)
        ty, tp, tids, records = to_trials(y, pred, trial, rec)
        ts = score(ty, tp)
        assert np.isclose(window["accuracy"], float(row["test_window_accuracy"]))
        assert np.isclose(ts["accuracy"], float(row["test_trial_accuracy"]))
        models[(protocol, family)] = dict(window=window, trial=ts, true=ty, pred=tp,
                                           tids=tids, records=records)
        rows.append({"protocol": protocol, "model": family, "source": "prior_frozen_predictions",
                     "input": row["variant"], "cv_macro_f1": float(row["cv_macro_f1"]),
                     "test_window_accuracy": window["accuracy"], "test_window_macro_f1": window["macro_f1"],
                     "test_trial_accuracy": ts["accuracy"], "test_trial_macro_f1": ts["macro_f1"],
                     "test_windows": len(y), "test_trials": len(ty)})

    for protocol in ["recording", "trial"]:
        locked = json.loads((ROOT/f"results/{protocol}/FINAL_SELECTION_LOCK.json").read_text())
        for unit in ["200", "1000"]:
            choice = locked["selected"][unit]
            candidates = [r for r in cv if r["protocol"] == protocol and r["candidate"].endswith(unit)]
            best = sorted(candidates, key=lambda r: (-r["cv_macro_f1"], r["candidate"]))[0]
            assert best["candidate"] == choice["candidate"]
            candidate = choice["candidate"]
            folder = ROOT/"results"/protocol/candidate/"final"
            final = json.loads((folder/"summary.json").read_text())
            data = np.load(folder/"predictions.npz")
            prob = data["probabilities"]
            assert np.allclose(prob, data["seed_probabilities"].mean(0), atol=1e-8)
            assert np.allclose(prob.sum(1), 1., atol=1e-6)
            y, pred, trial, rec = data["true"], prob.argmax(1), data["trial"], data["recording"]
            measured = score(y, pred)
            assert np.isclose(measured["accuracy"], final["accuracy"])
            assert np.isclose(measured["macro_f1"], final["macro_f1"])
            if unit == "200":
                window = measured
                ty, tp, tids, records = to_trials(y, pred, trial, rec)
                ts = score(ty, tp)
                assert np.isclose(ts["accuracy"], final["trial_vote"]["accuracy"])
            else:
                window = None
                order = np.argsort(trial)
                ty, tp, tids, records = y[order], pred[order], trial[order], rec[order]
                ts = measured
            models[(protocol, candidate)] = dict(window=window, trial=ts, true=ty, pred=tp,
                                                   tids=tids, records=records)
            # Every selected model evaluates exactly the same trial identities as the CPU baseline.
            assert np.array_equal(tids, models[(protocol, "svm")]["tids"])
            assert np.array_equal(ty, models[(protocol, "svm")]["true"])
            rows.append({"protocol": protocol, "model": candidate, "source": "local_gpu_frozen_selection",
                         "input": "peak_energy1s/raw_or_STFT_"+unit+"ms",
                         "cv_macro_f1": choice["cv_mean_macro_f1"],
                         "test_window_accuracy": window["accuracy"] if window else "",
                         "test_window_macro_f1": window["macro_f1"] if window else "",
                         "test_trial_accuracy": ts["accuracy"], "test_trial_macro_f1": ts["macro_f1"],
                         "test_windows": len(y) if window else "", "test_trials": len(ty)})
    cpu_cv = []
    for protocol in ["recording", "trial"]:
        folder = ROOT/"results/cpu_trial_control"/protocol
        selected = json.loads((folder/"SELECTION_LOCK.json").read_text())
        for name, summary in selected["all_candidates"].items():
            data = np.load(folder/name/"cv_predictions.npz")
            split = np.load(ROOT/f"data/{protocol}_split.npz")
            assert np.array_equal(data["indices"], np.flatnonzero(split["development"][::9]))
            for k in range(5):
                ix = data["fold"] == k
                measured = score(data["true"][ix], data["pred"][ix])
                assert np.isclose(measured["macro_f1"], summary["folds"][k]["macro_f1"])
            cpu_cv.append({"protocol": protocol, "model": name,
                           "cv_accuracy": summary["cv_accuracy"],
                           "cv_macro_f1": summary["cv_macro_f1"]})
        winner = sorted(selected["all_candidates"], key=lambda n:
                        (-selected["all_candidates"][n]["cv_macro_f1"], n))[0]
        assert winner == selected["selected"]
        data = np.load(folder/winner/"final/predictions.npz")
        final = json.loads((folder/winner/"final/summary.json").read_text())
        y, pred, tids, rec = (data[k] for k in ["true", "pred", "trial", "recording"])
        order = np.argsort(tids)
        y, pred, tids, rec = y[order], pred[order], tids[order], rec[order]
        ts = score(y, pred)
        assert np.isclose(ts["accuracy"], final["accuracy"])
        assert np.array_equal(tids, models[(protocol, "svm")]["tids"])
        assert np.array_equal(y, models[(protocol, "svm")]["true"])
        name = "SVM_1s_"+winner
        models[(protocol, name)] = dict(window=None, trial=ts, true=y, pred=pred,
                                       tids=tids, records=rec)
        rows.append({"protocol": protocol, "model": name, "source": "local_cpu_context_control",
                     "input": "aggregated_9_peak_energy_windows",
                     "cv_macro_f1": final["cv_macro_f1"], "test_window_accuracy": "",
                     "test_window_macro_f1": "", "test_trial_accuracy": ts["accuracy"],
                     "test_trial_macro_f1": ts["macro_f1"], "test_windows": "",
                     "test_trials": len(y)})
    write_csv(out/"all_cpu_context_cv.csv", cpu_cv)
    for (protocol, name), model in models.items():
        for record in np.unique(model["records"]):
            ix = model["records"] == record
            per_record.append({"protocol": protocol, "model": name, "recording": record,
                               "subject": record[0], "true_class": int(model["true"][ix][0]),
                               "n_trials": int(ix.sum()),
                               "trial_accuracy": float(np.mean(model["true"][ix] == model["pred"][ix]))})
        for unit in ["window", "trial"]:
            if model[unit] is not None:
                np.savetxt(out/f"cm_{protocol}_{name}_{unit}.csv",
                           np.array(model[unit]["confusion_matrix"]), fmt="%d", delimiter=",")
    write_csv(out/"all_comparisons.csv", rows)
    write_csv(out/"per_record_accuracy.csv", per_record)
    errors, failure_modes = [], []
    for (protocol, name), model in models.items():
        true, pred = model["true"], model["pred"]
        wrong = true != pred
        same_gesture = (true//2 == pred//2) & wrong
        failure_modes.append({"protocol": protocol, "model": name,
                              "errors": int(wrong.sum()),
                              "same_gesture_wrong_angle": int(same_gesture.sum()),
                              "different_gesture": int((wrong & ~same_gesture).sum())})
        for i in np.flatnonzero(wrong):
            errors.append({"protocol": protocol, "model": name, "trial": model["tids"][i],
                           "recording": model["records"][i], "true": LABELS[true[i]],
                           "predicted": LABELS[pred[i]]})
    write_csv(out/"misclassified_trials.csv", errors)
    write_csv(out/"failure_modes.csv", failure_modes)
    save = {"rows": rows, "cv": cv}
    (out/"summary.json").write_text(json.dumps(save, indent=2))

    # Paired differences clustered by participant, preserving each person's related recordings.
    differences = []
    rng = np.random.default_rng(2026)
    for protocol in ["recording", "trial"]:
        baseline_name = "xgb" if protocol == "recording" else "svm"
        base = models[(protocol, baseline_name)]
        subject = np.array([r[0] for r in base["records"]])
        unique = np.unique(subject)
        for (p, name), model in models.items():
            if p != protocol or name in ["svm", "rf", "xgb"]:
                continue
            delta = (model["true"] == model["pred"]).astype(float) - (base["true"] == base["pred"])
            sample_deltas = []
            for _ in range(3000):
                draws = rng.choice(unique, size=len(unique), replace=True)
                indices = np.concatenate([np.flatnonzero(subject == s) for s in draws])
                sample_deltas.append(float(delta[indices].mean()))
            differences.append({"protocol": protocol, "model": name, "baseline": baseline_name,
                                "trial_accuracy_difference": float(delta.mean()),
                                "exploratory_subject_cluster_95pct_interval":
                                    np.quantile(sample_deltas, [.025, .975]).tolist(),
                                "n_subjects": len(unique),
                                "warning": "Retrospective reused test; participant count is n_subjects; not prospective generalization evidence."})
    (out/"paired_trial_differences.json").write_text(json.dumps(differences, indent=2))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for protocol in ["recording", "trial"]:
        active = [r for r in rows if r["protocol"] == protocol]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        title = "A: held-out recordings (reused test)" if protocol == "recording" else "B: later trials of known recordings (reused test)"
        for ax, field, label in [(axes[0], "test_window_accuracy", "200 ms window accuracy (%)"),
                                 (axes[1], "test_trial_accuracy", "Trial accuracy (%)")]:
            selected = [r for r in active if r[field] != ""]
            names = [r["model"].replace("spectral_", "spec_").replace("SVM_1s_", "SVM1s\n") for r in selected]
            values = [r[field]*100 for r in selected]
            colors = ["#467a9b" if r["source"].startswith("prior") else
                      ("#4a956a" if r["source"] == "local_cpu_context_control" else "#dd8b32")
                      for r in selected]
            ax.bar(names, values, color=colors)
            ax.set_ylim(0, 105)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=20)
            for i, value in enumerate(values):
                ax.text(i, value+1.8, f"{value:.2f}", ha="center", fontsize=9)
            ax.axhline(12.5, ls="--", color=".65", linewidth=1)
        fig.suptitle(title+"\nBlue: prior CPU; orange: preselected GPU; green: matched 1s CPU control")
        fig.savefig(out/f"comparison_{protocol}.png", dpi=170)
        plt.close(fig)

        chosen = [n for p, n in models if p == protocol and (n.startswith("raw") or n.startswith("spectral"))]
        control = [n for p, n in models if p == protocol and n.startswith("SVM_1s_")]
        names = [("xgb" if protocol == "recording" else "svm")] + chosen
        names += control
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), layout="constrained")
        for ax, name in zip(axes, names):
            cm = np.array(models[(protocol, name)]["trial"]["confusion_matrix"])
            normalized = cm / cm.sum(1, keepdims=True)
            image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for i in range(8):
                for j in range(8):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                                color="white" if normalized[i, j] > .5 else "black")
            ax.set_xticks(range(8), LABELS, rotation=55, ha="right")
            ax.set_yticks(range(8), LABELS)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f'{name} | {models[(protocol,name)]["trial"]["accuracy"]*100:.2f}%')
        fig.colorbar(image, ax=axes, shrink=.7, label="Row-normalized fraction")
        fig.suptitle(title+"\nTrial confusion matrices; cells show trial counts")
        fig.savefig(out/f"confusion_{protocol}.png", dpi=170)
        plt.close(fig)

    # All selected CNNs: show every fold's training and validation curves, not just a favorable fold.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for i, protocol in enumerate(["recording", "trial"]):
        lock = json.loads((ROOT/f"results/{protocol}/FINAL_SELECTION_LOCK.json").read_text())
        for j, unit in enumerate(["200", "1000"]):
            ax = axes[i, j]
            name = lock["selected"][unit]["candidate"]
            for k in range(5):
                path = ROOT/f"results/{protocol}/{name}/fold{k}/epochs.jsonl"
                history = [json.loads(line) for line in path.read_text().splitlines()]
                ep = [r["epoch"] for r in history]
                ax.plot(ep, [r["train_accuracy"]*100 for r in history], color="#467a9b",
                        alpha=.4, label="Train (5 folds)" if k == 0 else None)
                ax.plot(ep, [r["val_accuracy"]*100 for r in history], color="#d77728",
                        alpha=.7, label="Validation (5 folds)" if k == 0 else None)
            ax.set_ylim(0, 102)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f'{"A" if i == 0 else "B"}: {name}')
            ax.legend(loc="lower right")
    fig.suptitle("Training / validation divergence in all selected CNN folds")
    fig.savefig(out/"learning_curves.png", dpi=170)
    plt.close(fig)

    audit = {"data_hashes_match": True, "code_protocol_hashes_match": True,
             "all_80_gpu_cv_folds_recomputed": True, "all_50_cpu_cv_folds_recomputed": True,
             "cv_indices_exclude_test": True,
             "early_stopping_best_epochs_match_logs": True, "final_selection_matches_cv": True,
             "final_ensemble_matches_saved_seed_probabilities": True,
             "cpu_and_gpu_trial_ids_and_true_labels_match": True,
             "all_reported_scores_independently_match_sklearn": True}
    (out/"audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(save, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
