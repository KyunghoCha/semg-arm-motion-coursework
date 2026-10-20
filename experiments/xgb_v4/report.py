"""Build full comparisons and public artifacts from the frozen XGBoost results."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/semg_xgb_mpl")
import csv, json, shutil
import importlib.metadata
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_experiment import *

FIG = OUT / "figures"
REPORT = OUT / "reports"
LABELS = ["주먹30", "주먹60", "펴기30", "펴기60", "회내30", "회내60", "회외30", "회외60"]
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "figure.dpi": 140,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})


def aggregate(paths):
    rows = [json.loads(p.read_text()) for p in paths]
    a = [np.load(p.parent / "predictions.npz") for p in paths]
    y, pred = np.concatenate([v["true"] for v in a]), np.concatenate([v["pred"] for v in a])
    return dict(mean_accuracy=float(np.mean([r["accuracy"] for r in rows])),
                sd_accuracy=float(np.std([r["accuracy"] for r in rows], ddof=1)),
                mean_macro_f1=float(np.mean([r["macro_f1"] for r in rows])),
                mean_gesture_accuracy=float(np.mean([r["gesture_accuracy"] for r in rows])),
                mean_angle_accuracy=float(np.mean([r["angle_accuracy"] for r in rows])),
                mean_train_accuracy=float(np.mean([r["train_accuracy"] for r in rows])),
                fit_seconds=float(sum(r["fit_seconds"] for r in rows)),
                same_gesture_angle_errors=int(np.sum((y != pred) & (y // 2 == pred // 2))),
                all_errors=int(np.sum(y != pred)), pooled=score(y, pred), folds=rows)


def main():
    freeze()
    FIG.mkdir(exist_ok=True)
    if not (REPORT / "environment.json").exists():
        build = xgb.build_info()
        build.pop("libxgboost", None)
        write(REPORT / "environment.json", dict(python=platform.python_version(),
              xgboost=xgb.__version__, numpy=np.__version__, build_info=build,
              nvidia_nccl_cu12=importlib.metadata.version("nvidia-nccl-cu12"),
              timestamp=stamp(), provenance="reproduction runtime; original install audit is in the published artifacts",
              base_environment_preservation_measured=False))
    lock = json.loads((OUT / "results/SELECTION_LOCK.json").read_text())
    best = lock["best_development"]
    dev = {name: aggregate(sorted((OUT / f"results/screen/{name}").glob("fold*/summary.json"))) for name in CONFIG}
    nested = aggregate(sorted((OUT / "results/outer").glob("fold*/summary.json")))
    diag = {kind: aggregate(sorted((OUT / f"results/diagnostics/{kind}").glob("*/summary.json")))
            for kind in ("record_seed20261022", "record_seed20261023", "subject")}
    fits = [json.loads(p.read_text()) for p in (OUT / "results").glob("**/summary.json")]
    samples = [v for r in fits for v in r["resources"]]
    resources = dict(completed_fit_calls=len(fits), screen_fits=120, inner_fits=360, outer_fits=5,
                     diagnostic_fits=18, representative_fits=1,
                     summed_fit_seconds=sum(r["fit_seconds"] for r in fits),
                     first_completed_at=min(r["timestamp"] for r in fits),
                     last_completed_at=max(r["timestamp"] for r in fits),
                     sampled_max_temperature_c=max(v["temperature_c"] for v in samples),
                     sampled_max_device_memory_mib=max(v["device_memory_mib"] for v in samples),
                     failure_count=len(list((OUT / "results").glob("**/failure.json"))),
                     synthetic_predata_failure_resolved=(REPORT / "self_check_failure_resolved.json").exists(), A_B_test_evaluations=0,
                     sampling="before/after each fit; not continuous peak measurement")
    result = dict(best_development=best, development=dev, nested=nested,
                  nested_selected=[r["name"] for r in lock["outer"]], diagnostics=diag, resources=resources)
    write(REPORT / "all_comparisons.json", result)
    write(REPORT / "resources.json", resources)
    keys = [k for k in dev[best] if k not in ("pooled", "folds")]
    with (REPORT / "development_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", *keys]); w.writeheader()
        w.writerows(dict(name=n, **{k: dev[n][k] for k in keys}) for n in CONFIG)

    # Full factorial structure/depth contrasts, same folds in every panel.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    for ax, feature, title in zip(axes.flat, ("legacy", "robust", "fusion", "relative"), ("기존 요약", "분포 요약", "결합", "상대량")):
        for structure, color, label in [("joint", "#287ca1", "8클래스 직접"), ("independent", "#d17b26", "동작·각도 독립")]:
            rows = [dev[f"{feature}_{structure}_d{d}"] for d in (2, 4, 6)]
            ax.errorbar([2, 4, 6], [r["mean_accuracy"]*100 for r in rows],
                        yerr=[r["sd_accuracy"]*100 for r in rows], fmt="o-", capsize=3, color=color, label=label)
        ax.set(title=title, xlabel="트리 깊이", xticks=[2, 4, 6], ylim=(0, 75))
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("개발 기록 5폴드 시행 정확도 (%)")
    axes[1, 0].set_ylabel("개발 기록 5폴드 시행 정확도 (%)")
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "factor_comparison.png"); plt.close(fig)

    # Curves are displayed after all selections, never used for early stopping.
    curves = [json.loads((OUT / f"results/screen/{best}/fold{k}/curves.json").read_text()) for k in range(5)]
    fig, axes = plt.subplots(1, len(curves[0]), figsize=(6.2*len(curves[0]), 4), squeeze=False)
    for i, ax in enumerate(axes[0]):
        for split, color, label in [("train", "#287ca1", "학습(클래스 가중)"), ("validation", "#d17b26", "검증(무가중)")]:
            a = np.array([c[i][split]["mlogloss"] for c in curves])
            ax.plot(np.arange(1, 201), a.mean(0), color=color, label=label)
            ax.fill_between(np.arange(1, 201), a.mean(0)-a.std(0, ddof=1), a.mean(0)+a.std(0, ddof=1), color=color, alpha=.15)
        title = "8클래스" if len(curves[0]) == 1 else ("손동작" if i == 0 else "팔꿈치 각도")
        ax.set(title=f"개발 대표: {title}", xlabel="고정 boosting round", ylabel="로그 손실"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "learning_curves.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    for ax, value, title in [(axes[0], dev[best], "개발 대표(선택에 사용)"), (axes[1], nested, "24종 선택 절차(중첩 외부)")]:
        cm = np.array(value["pooled"]["confusion_matrix"])
        frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > .55 else "#17233a")
        ax.set(title=title, xticks=range(8), yticks=range(8), xticklabels=LABELS, yticklabels=LABELS, xlabel="예측", ylabel="정답")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "confusion.png"); plt.close(fig)

    prior = json.loads((TASK / "step2_ml_v3/reports/split_diagnostics.json").read_text())
    lda = next(r for r in prior if r["model"] == "fusion_ind_lda" and r["diagnostic"] == "subject")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for vals, label, color, offset in [(lda["folds"], "앞선 독립 LDA", "#287ca1", -.18), (diag["subject"]["folds"], "고정 XGBoost 대표", "#d17b26", .18)]:
        ax.bar(np.arange(8)+offset, [r["accuracy"]*100 for r in vals], width=.36, color=color, label=label)
    ax.set(xticks=range(8), xticklabels=list("abcdefgh"), xlabel="학습에서 제외한 참가자", ylabel="개발 시행 정확도 (%)", ylim=(0, 100))
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "person_sensitivity.png"); plt.close(fig)

    rows = []
    log = ["# XGBoost 24종 비교 기록", "", "모든 후보는 고정한 A 개발 기록 5폴드에서 비교했다. 기존 A/B 시험은 사용하지 않았다.", ""]
    for name, cfg in CONFIG.items():
        r = dev[name]
        reference = f"{cfg['feature']}_joint_d{cfg['depth']}" if cfg["structure"] == "independent" else f"{cfg['feature']}_joint_d2"
        delta = r["mean_macro_f1"] - dev[reference]["mean_macro_f1"]
        reason = "동작·각도별 표본을 모아 경계를 학습하면 기록 변화에 대응하는가" if cfg["structure"] == "independent" else "트리 깊이에 따른 복잡도·과적합 변화는 어떠한가"
        decision = "XGBoost 개발 대표로 선택" if name == best else "비교 근거로 보존; 대표 미선택"
        log += [f"## `{name}`", "", f"- 관찰·가설: {reason}.",
                f"- 변경·고정: 특징 {cfg['feature']}, 구조 {cfg['structure']}, 깊이 {cfg['depth']}. 1초 입력·기록 분할·200라운드·나머지 정규화 설정 고정.",
                f"- 비교 기준: `{reference}`. 결과·설정은 [fold0](artifacts/results/screen/{name}/fold0/summary.json).",
                f"- 결과: 정확도 {r['mean_accuracy']*100:.2f}±{r['sd_accuracy']*100:.2f}%, F1 {r['mean_macro_f1']:.4f}, 각도 {r['mean_angle_accuracy']*100:.2f}%, 학습 {r['mean_train_accuracy']*100:.2f}%; 5회 합계 {r['fit_seconds']:.2f}초.",
                f"- 해석·선택: 기준 대비 F1 차이는 {delta:+.4f}였다. {decision}.",
                "- 후속: 사전 고정한 중첩 선택 절차와 대표의 분할 민감도 진단으로 확인했다. 그 결과로 후보를 다시 조정하지 않았다.", ""]
        rows.append(f"| {name} | {r['mean_accuracy']*100:.2f}±{r['sd_accuracy']*100:.2f} | {r['mean_macro_f1']:.4f} | {r['mean_gesture_accuracy']*100:.2f} | {r['mean_angle_accuracy']*100:.2f} | {r['mean_train_accuracy']*100:.2f} | {r['fit_seconds']:.2f} |")
    (HERE / "EXPERIMENT_LOG.md").write_text("\n".join(log))
    (REPORT / "candidate_table.md").write_text("| 후보 | 정확도 평균±SD (%) | macro-F1 | 동작 (%) | 각도 (%) | 학습 (%) | 5회 학습 (초) |\n|---|---:|---:|---:|---:|---:|---:|\n"+"\n".join(rows)+"\n")

    # Explicit publication allowlist: no features, models, installation paths or raw lectures.
    public = HERE / "artifacts"
    for dirname in ("results", "figures", "logs"):
        for src in (OUT / dirname).rglob("*"):
            if not src.is_file() or src.suffix not in (".json", ".npz", ".png", ".log"):
                continue
            if dirname == "logs" and src.stem not in ("screen", "inner", "select", "outer", "diagnostics", "representative"):
                continue
            dest = public / src.relative_to(OUT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
    for name in ("all_comparisons.json", "resources.json", "development_all.csv", "candidate_table.md", "verification.json", "self_check.json", "self_check_failure_resolved.json", "environment.json"):
        src = REPORT / name
        if not src.exists() and name == "self_check_failure_resolved.json":
            continue  # This historical failure need not recur in a corrected reproduction.
        dest = public / "reports" / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
    shutil.copy2(DATA / "manifest.json", public / "reports/input_manifest.json")
    shutil.copy2(DATA / "splits.npz", public / "reports/development_splits.npz")
    shutil.copy2(TASK / "step2_ml_v3/results/diagnostic_splits.npz", public / "reports/diagnostic_splits.npz")
    print(json.dumps({"best": best, "development_accuracy": dev[best]["mean_accuracy"],
                      "nested_accuracy": nested["mean_accuracy"], "resources": resources}, indent=2))


if __name__ == "__main__":
    main()
