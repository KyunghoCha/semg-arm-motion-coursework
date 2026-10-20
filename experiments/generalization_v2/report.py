"""Aggregate immutable predictions, audit splits/hashes, and render report figures."""
import csv, json, platform, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common import *

FIG = OUT/"figures"
REPORT = OUT/"reports"
LABELS = ["주먹30", "주먹60", "펴기30", "펴기60", "회내30", "회내60", "회외30", "회외60"]
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False, "figure.dpi": 130})


def csvsave(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def cmplot(matrices, titles, filename):
    fig, axes = plt.subplots(1, len(matrices), figsize=(6*len(matrices), 5.2), squeeze=False)
    for ax, cm, title in zip(axes[0], matrices, titles):
        cm = np.asarray(cm)
        ax.imshow(cm/cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j]/cm[i].sum() > .55 else "#17233a")
        ax.set(xticks=range(8), yticks=range(8), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="예측", ylabel="정답", title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(FIG/filename); plt.close(fig)


def folds_pred(root, n=5):
    result = []
    for k in range(n):
        a = np.load(root/f"fold{k}/validation.npz")
        pred = a["probabilities"].argmax(1) if "probabilities" in a else a["pred"]
        result.append((a["indices"], a["true"], pred))
    return result


def main():
    FIG.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    s = splits()
    lock = json.loads((OUT/"results/ADAPTIVE_SELECTION_LOCK.json").read_text())
    names = list(lock["single_candidates"])
    screens = []; diagnostics = {}
    for name in names:
        r = summarize(OUT/f"results/screen/{name}")
        folds = folds_pred(OUT/f"results/screen/{name}")
        y = np.concatenate([f[1] for f in folds]); p = np.concatenate([f[2] for f in folds])
        diag = score(y, p)
        diag.update(errors=int(np.sum(y != p)), same_gesture_angle_errors=int(np.sum((y != p)&(y//2 == p//2))))
        diagnostics[name] = diag
        row = {"model": name, "mean_accuracy": r["accuracy"], "sd_accuracy": r["accuracy_sd"],
               "mean_macro_f1": r["macro_f1"], "epochs": r["epochs"]}
        if name not in CPU:
            atbest = []
            for k, summary in enumerate(r["folds"]):
                log = [json.loads(l) for l in (OUT/f"results/screen/{name}/fold{k}/epochs.jsonl").read_text().splitlines()]
                atbest.append(log[summary["best_epoch"]-1]["train_accuracy"])
            row["mean_train_accuracy_at_selected_epoch"] = float(np.mean(atbest))
        else:
            row["mean_train_accuracy_at_selected_epoch"] = None
        screens.append(row)
    csvsave(REPORT/"development_comparison.csv", screens)
    write(REPORT/"development_diagnostics.json", diagnostics)

    nestedlock = json.loads((OUT/"results/NESTED_SELECTION_LOCK.json").read_text())
    nested = {}; outertable = []
    for kind in ["baseline", "selected"]:
        trues, preds, indices, folds = [], [], [], []
        for k, entry in enumerate(nestedlock["folds"]):
            name = "spectral1s" if kind == "baseline" else entry["selected"]
            folder = OUT/f"results/nested_outer/outer{k}/{name}"
            a = np.load(folder/"validation.npz")
            r = json.loads((folder/"summary.json").read_text())
            trues.extend(a["true"]); preds.extend(a["probabilities"].argmax(1)); indices.extend(a["indices"])
            folds.append(r)
            outertable.append({"fold": k, "procedure": kind, "model": name, "epochs": r["best_epoch"],
                               "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]})
        nested[kind] = {"pooled": score(np.array(trues), np.array(preds)),
                        "mean_accuracy": float(np.mean([r["accuracy"] for r in folds])),
                        "sd_accuracy": float(np.std([r["accuracy"] for r in folds], ddof=1)),
                        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in folds])),
                        "folds": folds}
        np.savez_compressed(REPORT/f"nested_{kind}_predictions.npz", indices=indices, true=trues, pred=preds)
    csvsave(REPORT/"nested_outer_comparison.csv", outertable)

    final = {}; final_predictions = {}
    for name in lock["evaluate"]:
        folder = OUT/f"results/final/{name}"
        arrays = [np.load(p) for p in sorted(folder.glob("seed*/validation.npz"))]
        assert len(arrays) == (1 if name in CPU else 2)
        assert all(np.array_equal(a["indices"], np.flatnonzero(s["test"])) for a in arrays)
        if name in CPU:
            pred = arrays[0]["pred"]; probabilities = None
        else:
            probabilities = np.mean([a["probabilities"] for a in arrays], axis=0)
            pred = probabilities.argmax(1)
        y = arrays[0]["true"]
        r = score(y, pred)
        r.update(errors=int(np.sum(y != pred)), same_gesture_angle_errors=int(np.sum((y != pred)&(y//2 == pred//2))))
        final[name] = r
        final_predictions[name] = pred
        np.savez_compressed(folder/"aggregate_predictions.npz", indices=arrays[0]["indices"], true=y,
                            pred=pred, **({} if probabilities is None else {"probabilities": probabilities}))
        write(folder/"aggregate_metrics.json", r)
    selected = lock["selected"]
    assert selected != "ensemble", "This completed run selected a single architecture"
    y = s["y"][s["test"]]; records = s["recording"][s["test"]]; subjects = s["subject"][s["test"]]
    recordrows = []
    for rec in np.unique(records):
        mask = records == rec
        recordrows.append({"record": rec, "subject": str(subjects[mask][0]), "class": int(y[mask][0]),
                          "n": int(mask.sum()), "baseline_accuracy": float(np.mean(final_predictions["spectral1s"][mask] == y[mask])),
                          "selected_accuracy": float(np.mean(final_predictions[selected][mask] == y[mask]))})
    csvsave(REPORT/"final_per_record.csv", recordrows)
    difference = (final_predictions[selected] == y).astype(float) - (final_predictions["spectral1s"] == y)
    people = np.unique(subjects)
    sums = np.array([difference[subjects == p].sum() for p in people])
    ns = np.array([(subjects == p).sum() for p in people])
    draws = np.random.default_rng(20261021).integers(0, len(people), (10000, len(people)))
    bootstrap = sums[draws].sum(1)/ns[draws].sum(1)
    uncertainty = {"selected_minus_baseline": float(difference.mean()), "unit": "participant clusters",
                   "participants": len(people), "records": len(recordrows), "bootstrap_seed": 20261021,
                   "draws": 10000, "percentile_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
                   "scope": "descriptive uncertainty on reused test; few participants and selection history limit inference"}

    classroom_rows = json.loads((OUT/"results/classroom/all_results.json").read_text())
    classroom = []; paired = []
    for assembly in ["window_vote", "trial_mean"]:
        for model in ["SVM", "RF"]:
            rows = [r for r in classroom_rows if r["assembly"] == assembly and r["model"] == model]
            cv = [r for r in rows if r["split"] != "holdout"]
            holdout = [r for r in rows if r["split"] == "holdout"][0]
            for unit in (["window", "trial"] if assembly == "window_vote" else ["trial"]):
                classroom.append({"assembly": assembly, "model": model, "unit": unit,
                    "cv_mean_accuracy": float(np.mean([r[unit]["accuracy"] for r in cv])),
                    "cv_sd_accuracy": float(np.std([r[unit]["accuracy"] for r in cv], ddof=1)),
                    "cv_macro_f1": float(np.mean([r[unit]["macro_f1"] for r in cv])),
                    "holdout_accuracy": holdout[unit]["accuracy"], "holdout_macro_f1": holdout[unit]["macro_f1"],
                    "mean_cv_fit_seconds": float(np.mean([r["fit_seconds"] for r in cv])),
                    "holdout_single_input_ms": holdout["single_input_inference_ms"]})
        svm = [r["trial"]["accuracy"] for r in classroom_rows if r["assembly"] == assembly and r["model"] == "SVM" and r["split"] != "holdout"]
        rf = [r["trial"]["accuracy"] for r in classroom_rows if r["assembly"] == assembly and r["model"] == "RF" and r["split"] != "holdout"]
        t = stats.ttest_rel(rf, svm)
        paired.append({"assembly": assembly, "unit": "trial", "n_folds": 5, "rf_minus_svm": float(np.mean(rf)-np.mean(svm)),
                       "t": float(t.statistic), "p_two_sided_unadjusted": float(t.pvalue),
                       "scope": "descriptive paired folds; overlapping training sets and repeated participants; not confirmatory"})
    csvsave(REPORT/"classroom_comparison.csv", classroom)

    # Verify predictions, frozen source, split integrity and baseline replication.
    verified = {}
    for name, digest in json.loads((DATA/"sha256.json").read_text()).items():
        assert sha(DATA/name) == digest
    for name, digest in lock["source_hashes"].items():
        assert sha(HERE/name) == digest
    for name in names:
        for k, (idx, true, pred) in enumerate(folds_pred(OUT/f"results/screen/{name}")):
            assert np.array_equal(idx, np.flatnonzero(s["development"] & (s["fold"] == k)))
            assert np.array_equal(true, s["y"][idx])
            assert_groups(s, s["development"] & (s["fold"] != k), s["development"] & (s["fold"] == k))
    for k in range(5):
        pool = s["development"] & (s["fold"] != k)
        for j in range(3):
            assert_groups(s, pool & (s[f"inner{k}"] != j), pool & (s[f"inner{k}"] == j))
    for c in range(5):
        tr, va = s["c_fold"] != c, s["c_fold"] == c
        assert set(s["trial"][tr]).isdisjoint(s["trial"][va])
    previous = np.load(OLD/"results/recording/spectral1000/final/predictions.npz")
    assert np.array_equal(previous["probabilities"].argmax(1), final_predictions["spectral1s"])
    baseline_probs = np.load(OUT/"results/final/spectral1s/aggregate_predictions.npz")["probabilities"]
    max_error = float(np.max(np.abs(previous["probabilities"]-baseline_probs)))
    verified.update(data_cache_hashes=True, frozen_training_source_hashes=True,
                    all_development_prediction_indices=True, all_grouped_splits=True,
                    classroom_trial_groups=True, baseline_predictions_exact=True,
                    baseline_probability_max_abs_error=max_error,
                    final_checkpoint_report="final_checkpoint_verification.json")
    assert lock["timestamp"] < min(r["timestamp"] for v in nested.values() for r in v["folds"])
    assert lock["timestamp"] < min(json.loads(p.read_text())["timestamp"] for p in (OUT/"results/final").glob("*/seed*/summary.json"))
    verified["choices_locked_before_outer_and_test"] = True
    write(REPORT/"verification.json", verified)
    resources = []
    for p in (OUT/"results").glob("**/epochs.jsonl"):
        resources.extend(json.loads(l) for l in p.read_text().splitlines())
    fitted = [json.loads(p.read_text()) for p in (OUT/"results").glob("**/summary.json")]
    neural = [r for r in fitted if "allocated_mib" in r]
    resource_summary = {"gpu": "NVIDIA TITAN Xp", "vram_mib": 12288, "driver": "580.178.04",
        "nvidia_smi_cuda_max": "13.0", "torch": "2.6.0+cu118", "torch_cuda_runtime": "11.8",
        "gpu_completed_fits": len(neural), "gpu_summed_fit_seconds": sum(r["seconds"] for r in neural),
        "max_temperature_c": max(r["gpu_temp_c"] for r in resources),
        "peak_torch_allocated_mib": max(r["allocated_mib"] for r in resources),
        "peak_device_used_mib": max(r["device_memory_mib"] for r in resources),
        "epochs_logged": len(resources), "all_model_fits": len(fitted),
        "installs": [], "cloud_compute": False}
    write(REPORT/"resources.json", resource_summary)
    result = {"timestamp": stamp(), "selected": selected, "development": screens,
              "nested": nested, "final": final, "classroom": classroom, "paired_classroom": paired,
              "uncertainty": uncertainty, "resources": resource_summary, "verification": verified}
    write(REPORT/"all_comparisons.json", result)

    # Figures: comparisons are separated by protocol.
    order = sorted(screens, key=lambda r: r["mean_macro_f1"])
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    yy = np.arange(len(order))
    ax.barh(yy, [r["mean_accuracy"]*100 for r in order], color=["#dc7633" if r["model"] == selected else "#36749c" for r in order])
    ax.errorbar([r["mean_accuracy"]*100 for r in order], yy,
                xerr=[r["sd_accuracy"]*100 for r in order], fmt="none", ecolor="#374151", capsize=2)
    ax.set(yticks=yy, yticklabels=[r["model"] for r in order], xlabel="A 개발 5겹 기록 분리: 시행 정확도 (%)", xlim=(0, 70))
    ax.grid(axis="x", alpha=.15); fig.tight_layout(); fig.savefig(FIG/"development_models.png"); plt.close(fig)
    cmplot([final["spectral1s"]["confusion_matrix"], final[selected]["confusion_matrix"]],
           ["기존 CNN: 재사용 A 시험", "개발에서 선택한 CNN: 재사용 A 시험"], "final_confusion.png")
    cmplot([nested["baseline"]["pooled"]["confusion_matrix"], nested["selected"]["pooled"]["confusion_matrix"]],
           ["기존 CNN: 중첩 외부 평가", "고정 16종 선택 절차: 중첩 외부 평가"], "nested_confusion.png")
    crows = [r for r in classroom_rows if r["assembly"] == "window_vote" and r["split"] == "holdout"]
    cmplot([r["trial"]["confusion_matrix"] for r in crows], ["C SVM: 시행 다수결", "C RF: 시행 다수결"], "classroom_confusion.png")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    for row, name in enumerate(["spectral1s", selected]):
        for k in range(5):
            log = [json.loads(l) for l in (OUT/f"results/screen/{name}/fold{k}/epochs.jsonl").read_text().splitlines()]
            ax = axes[row, k]; epochs = [r["epoch"] for r in log]
            ax.plot(epochs, [r["train_accuracy"]*100 for r in log], label="학습")
            ax.plot(epochs, [r["val_accuracy"]*100 for r in log], label="개발 검증")
            epoch = json.loads((OUT/f"results/screen/{name}/fold{k}/summary.json").read_text())["best_epoch"]
            ax.axvline(epoch, color="gray", linestyle="--", alpha=.6)
            ax.set(title=f"{'기존 CNN' if row == 0 else '선택 CNN'} / fold{k}", xlabel="epoch", ylim=(0, 103))
    axes[0, 0].set_ylabel("시행 정확도 (%)"); axes[1, 0].set_ylabel("시행 정확도 (%)")
    axes[0, 0].legend(fontsize=9); fig.tight_layout(); fig.savefig(FIG/"learning_curves.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    base = [r["accuracy"]*100 for r in nested["baseline"]["folds"]]
    chosen = [r["accuracy"]*100 for r in nested["selected"]["folds"]]
    for k in range(5):
        ax.plot([0, 1], [base[k], chosen[k]], marker="o", label=f"fold{k}")
    ax.set(xticks=[0, 1], xticklabels=["기존 CNN", "고정 16종 선택 절차"],
           ylabel="중첩 외부 시행 정확도 (%)", ylim=(0, 60))
    ax.legend(ncol=3); fig.tight_layout(); fig.savefig(FIG/"nested_folds.png"); plt.close(fig)
    # Aggregate channel observations use only A development records.
    raw = np.load(OLD/"data/raw1000.npy")
    rms = np.sqrt(np.mean(raw.astype("float64")**2, axis=-1))
    record_stats = []
    for rec in np.unique(s["recording"][s["development"]]):
        mask = s["recording"] == rec
        record_stats.append({"record": rec, "subject": str(s["subject"][mask][0]), "y": int(s["y"][mask][0]),
                             "rms": np.median(rms[mask], axis=0)})
    people = sorted({r["subject"] for r in record_stats})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    ratios = []
    for g, ax in enumerate(axes):
        for j, person in enumerate(people):
            rows = [r for r in record_stats if r["subject"] == person and r["y"]//2 == g]
            for r in rows:
                ax.scatter(r["rms"][3], r["rms"][4], marker="o" if r["y"]%2 == 0 else "^",
                           color=plt.cm.tab10(j), s=42)
            if len(rows) == 2:
                rows.sort(key=lambda r: r["y"])
                ax.plot([r["rms"][3] for r in rows], [r["rms"][4] for r in rows], color=plt.cm.tab10(j), alpha=.5)
                ratio = rows[1]["rms"][3:]/rows[0]["rms"][3:]
                ratios.append({"subject": person, "gesture": g, "ch4_ratio": float(ratio[0]), "ch5_ratio": float(ratio[1])})
        ax.set(xscale="log", yscale="log", xlabel="CH4 기록 RMS 중앙값", title=["주먹쥐기", "손펴기", "회내", "회외"][g])
    axes[0].set_ylabel("CH5 기록 RMS 중앙값")
    fig.tight_layout(); fig.savefig(FIG/"development_channel_distribution.png"); plt.close(fig)
    write(REPORT/"channel_observations.json", {
        "scope": "A development only; record median of30energy-selected1s trial RMS values",
        "pairs": len(ratios), "ch4_up_ch5_down": sum(r["ch4_ratio"] > 1 and r["ch5_ratio"] < 1 for r in ratios),
        "both_up": sum(r["ch4_ratio"] > 1 and r["ch5_ratio"] > 1 for r in ratios),
        "ch4_down_ch5_up": sum(r["ch4_ratio"] < 1 and r["ch5_ratio"] > 1 for r in ratios),
        "ratios": ratios})
    print(json.dumps({"nested": {k: {a: v[a] for a in ["mean_accuracy", "sd_accuracy", "mean_macro_f1"]} for k, v in nested.items()},
                      "final": {k: v["accuracy"] for k, v in final.items()}, "resources": resource_summary,
                      "uncertainty": uncertainty}, indent=2), flush=True)


if __name__ == "__main__":
    main()
