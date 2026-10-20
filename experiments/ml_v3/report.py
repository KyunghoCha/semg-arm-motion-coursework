"""Generate ML comparisons, per-experiment rationale, diagnostics and review artifacts."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/semg_ml_mpl")
import csv, json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import adaptive
from train import *

FIG = OUT/"figures"
REPORT = OUT/"reports"
LABELS = ["주먹30", "주먹60", "펴기30", "펴기60", "회내30", "회내60", "회외30", "회외60"]
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "figure.dpi": 140,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})


def csvsave(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def confusion(matrices, titles, name):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.6*len(matrices), 5), squeeze=False)
    for ax, cm, title in zip(axes[0], matrices, titles):
        cm = np.array(cm)
        ax.imshow(cm/cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(cm[i, j]), va="center", ha="center",
                        color="white" if cm[i, j]/cm[i].sum() > .55 else "#17233a", fontsize=9)
        ax.set(xticks=range(8), yticks=range(8), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="예측", ylabel="정답", title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(FIG/name); plt.close(fig)


def describe(name):
    if name in adaptive.CANDIDATES:
        c = adaptive.CANDIDATES[name]
        if name.startswith("mixed"):
            reference = "mixed_et_lda_logrms5" if name != "mixed_et_lda_logrms5" else "legacy_ind_lda"
            reason = "각도 분류의 입력을 제한하면 기록별 진폭 변동과 불필요한 변수를 줄일 수 있는가"
            change = f"손동작 {c['gesture_feature']}/{c['gesture_model']}, 각도 {c['angle_feature']}/{c['angle_model']}"
            if name == "mixed_et_et_lowref5s":
                reference = "mixed_et_lda_lowref5s"
        elif name == "fusion_ind_et6":
            reference = "fusion_et3_independent"; reason = "독립 분류기의 트리 말단을 키워 과적합을 줄일 수 있는가"; change = "leaf3→6"
        elif name == "fusion_ind_et3_unweighted":
            reference = "fusion_et3_independent"; reason = "클래스 가중치가 독립 구조의 이득을 설명하는가"; change = "balanced→무가중치"
        elif name == "fusion_ind_rf3":
            reference = "fusion_et3_independent"; reason = "독립 구조의 이득이 트리 분할 방법에 의존하는가"; change = "ExtraTrees→RF"
        else:
            suffix = {"et": "et3", "lda": "lda", "svm": "svm1"}[c["gesture_model"]]
            reference = f"{c['gesture_feature']}_{suffix}"
            reason = "8클래스를 직접 구분하는 대신 손동작과 각도별 표본을 모으면 경계 추정이 나아지는가"
            change = "같은 특징·분류기, 8클래스 직접 분류→4동작+2각도 독립 분류"
        return reference, reason, change, c["context"], "고정 1초/기록5겹; 5초 후보만 기준 문맥 추가"
    c = CONFIG[name]; feature = c["feature"]
    reasons = {
        "legacy": "기존 CPU 특징과 평가를 재현하는가",
        "robust": "중앙값과 IQR이 창별 변동을 보완하는가",
        "spectrum": "주파수 분포와 진폭을 분리하면 기록 차이를 줄이는가",
        "relations": "채널 상관·coherence·공분산이 개별 채널 특징을 보완하는가",
        "fusion": "시간·주파수·채널 관계가 상보적인가",
        "relative": "표본별 상대량이 전체 이득 변화의 영향을 줄이는가",
        "upper": "CH4·CH5만 남기면 어떤 정보가 유지되는가",
        "forearm": "CH1–CH3만 남기면 어떤 정보가 유지되는가"}
    reason = reasons[feature]
    reference = "legacy_svm1"
    change = f"특징={feature}, 분류기={c['model']}; 다요인 탐색"
    if c["structure"] != "joint":
        reference = "fusion_et3"; reason = "같은 특징과 트리에서 분류 구조만 바꿔 각도 구분을 개선하는가"
        change = f"joint→{c['structure']}"
    elif c["normalization"] == "robust":
        reference = "fusion_svm1"; reason = "극단값에 덜 민감한 스케일이 도움이 되는가"; change = "StandardScaler→RobustScaler"
    elif c["selection"] is not None:
        reference = "fusion_svm1" if c["model"] == "svm" else "fusion_et3"
        reason = "중복되거나 기록에 과적합한 변수를 줄일 수 있는가"
        change = f"특징 변환={c['selection']}; 나머지 설정 고정"
    elif name == "fusion_et3_unweighted":
        reference = "fusion_et3"; reason = "클래스 가중치의 영향은 어느 정도인가"; change = "balanced→무가중치"
    elif c["model"] == "svm":
        if c["C"] != 1:
            reference = f"{feature}_svm1"; reason = "SVM 규제 강도의 영향을 분리할 수 있는가"; change = f"C1→C{c['C']}"
        else:
            change = f"legacy→{feature}; SVM C1 고정"
    elif c["model"] == "et":
        reference = f"{feature}_et3" if c["leaf"] != 3 else f"{feature}_svm1"
        change = f"leaf3→{c['leaf']}" if c["leaf"] != 3 else "동일 특징: SVM→ExtraTrees"
    elif c["model"] == "lda":
        reference = f"{feature}_svm1"; change = "동일 특징: SVM→shrinkage LDA"
    elif c["model"] == "rf":
        reference = f"{feature}_rf1" if c["leaf"] == 6 else f"{feature}_et3"
        change = "RF leaf1→6" if c["leaf"] == 6 else "동일 특징: ExtraTrees→RF; 말단 설정은 config 참조"
    elif c["model"] == "hgb":
        reference = "fusion_hgb31" if c["leaves"] == 7 else "fusion_et3"
        change = "최대 말단31→7" if c["leaves"] == 7 else "동일 특징: ExtraTrees→Histogram GB"
    elif c["model"] == "linear":
        reference = "legacy_svm1"; change = "RBF SVM→LinearSVC"
    return reference, reason, change, "1s", "같은 1초·기록5겹·학습 폴드 통계"


def main():
    FIG.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    s = dict(np.load(DATA/"splits.npz"))
    selection = json.loads((OUT/"results/DEVELOPMENT_SELECTION_LOCK.json").read_text())
    nested_selection = json.loads((OUT/"results/SELECTION_LOCK.json").read_text())
    rows = []; pooled = {}
    for name in selection["rank"]:
        root = OUT/f"results/{'screen' if name in CONFIG else 'adaptive'}/{name}"
        values = summarize(root)
        fs = [np.load(root/f"fold{k}/predictions.npz") for k in range(5)]
        for k, a in enumerate(fs):
            assert np.array_equal(a["indices"], np.flatnonzero(s["fold"] == k))
            assert np.array_equal(a["true"], s["y"][a["indices"]])
        true = np.concatenate([a["true"] for a in fs]); pred = np.concatenate([a["pred"] for a in fs])
        pooled[name] = score(true, pred)
        pooled[name]["same_gesture_angle_errors"] = int(np.sum((true != pred)&(true//2 == pred//2)))
        pooled[name]["all_errors"] = int(np.sum(true != pred))
        context = "1s" if name in CONFIG else adaptive.CANDIDATES[name]["context"]
        rows.append({"model": name, "context": context, "stage": "fixed" if name in CONFIG else "adaptive",
                     "mean_accuracy": values["accuracy"], "sd_accuracy": values["accuracy_sd"],
                     "mean_macro_f1": values["macro_f1"],
                     "mean_gesture_accuracy": float(np.mean([r["gesture_accuracy"] for r in values["folds"]])),
                     "mean_angle_accuracy": float(np.mean([r["angle_accuracy"] for r in values["folds"]])),
                     "mean_train_accuracy": values["train_accuracy"], "five_fold_fit_seconds": values["fit_seconds"]})
    byname = {r["model"]: r for r in rows}
    csvsave(REPORT/"development_all.csv", rows)
    write(REPORT/"development_pooled.json", pooled)
    log = ["# 전체 실험 과정", "", "모든 수치는 A 개발 기록의 같은 5겹 검증이다. F1은 8클래스 macro-F1의 폴드 평균이다.",
           "채택은 개발 기준의 대표 선택을 의미한다. 기존 A/B 시험 성능의 채택·기각을 의미하지 않는다.",
           "원본 코드와 후보별 config·로그·예측은 각 링크에서 확인할 수 있다.", ""]
    decisions = []
    for name in [*CONFIG, *adaptive.CANDIDATES]:
        ref, reason, change, context, fixed = describe(name)
        r = byname[name]
        delta = r["mean_macro_f1"]-byname[ref]["mean_macro_f1"]
        selected = name == selection["best_1s"]
        state = "1초 개발 대표로 채택" if selected else "비교 근거로 보존; 1초 대표에는 미선택"
        if context == "5s":
            state = "추가 문맥의 별도 비교; 1초 대표와 구분"
        interpretation = ("기준보다 개발 평균 F1이 높았다" if delta > 1e-12 else
                          "기준보다 개발 평균 F1이 낮았다" if delta < -1e-12 else "기준과 같은 개발 평균 F1이었다")
        stage = "screen" if name in CONFIG else "adaptive"
        log += [f"## `{name}`", "", f"- 관찰·가설: {reason}.",
                f"- 변경·고정 조건: {change}. {fixed}.",
                f"- 비교 기준: `{ref}`. 상세 설정은 [fold0 기록](artifacts/results/{stage}/{name}/fold0/summary.json).",
                f"- 결과: 정확도 {100*r['mean_accuracy']:.2f} ± {100*r['sd_accuracy']:.2f}%, F1 {r['mean_macro_f1']:.4f}, "
                f"각도 {100*r['mean_angle_accuracy']:.2f}%, 5회 fit 합계 {r['five_fold_fit_seconds']:.2f}초.",
                f"- 해석·결정: {interpretation}(차이 {delta:+.4f}). {state}.",
                "- 다음 선택: 고정한 검증·민감도 진단으로 확인하며 이 평가 후 후보를 조정하지 않는다.", ""]
        decisions.append({"model": name, "reference": ref, "hypothesis": reason, "changed": change,
                          "fixed": fixed, "macro_f1_difference": delta, "decision": state})
    (HERE/"EXPERIMENT_LOG.md").write_text("\n".join(log))
    write(REPORT/"experiment_decisions.json", decisions)

    nested = {}; outer_rows = []
    for procedure in ["baseline", "single", "ensemble"]:
        true_all, pred_all, indices_all, folds = [], [], [], []
        for k, item in enumerate(nested_selection["outer"]):
            names = ["legacy_svm1"] if procedure == "baseline" else [item["single"]] if procedure == "single" else item["ensemble"]
            fs = [np.load(OUT/f"results/outer/outer{k}/{n}/predictions.npz") for n in names]
            assert all(np.array_equal(a["indices"], fs[0]["indices"]) for a in fs)
            pred = fs[0]["pred"] if procedure != "ensemble" else np.mean([a["probabilities"] for a in fs], axis=0).argmax(1)
            metrics = score(fs[0]["true"], pred); folds.append(metrics)
            outer_rows.append({"procedure": procedure, "fold": k, "models": "+".join(names),
                               "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            true_all.extend(fs[0]["true"]); pred_all.extend(pred); indices_all.extend(fs[0]["indices"])
        nested[procedure] = {"mean_accuracy": float(np.mean([a["accuracy"] for a in folds])),
                             "sd_accuracy": float(np.std([a["accuracy"] for a in folds], ddof=1)),
                             "mean_macro_f1": float(np.mean([a["macro_f1"] for a in folds])),
                             "pooled": score(np.array(true_all), np.array(pred_all)), "folds": folds}
        np.savez_compressed(REPORT/f"nested_{procedure}.npz", indices=indices_all, true=true_all, pred=pred_all)
    csvsave(REPORT/"nested_outer.csv", outer_rows)
    write(REPORT/"nested_comparison.json", nested)

    diagnostic_lock = json.loads((OUT/"results/DIAGNOSTICS_FROZEN.json").read_text())
    diagnostic = []; detailed = []
    for name in diagnostic_lock["models"]:
        for kind, count in [("record_seed20261022", 5), ("record_seed20261023", 5), ("subject", 8)]:
            paths = sorted((OUT/f"results/diagnostics/{name}/{kind}").glob("*/summary.json"))
            assert len(paths) == count
            values = [json.loads(p.read_text()) for p in paths]
            arrays = [np.load(p.parent/"predictions.npz") for p in paths]
            for path, val, arr in zip(paths, values, arrays):
                assert diagnostic_lock["timestamp"] < val["timestamp"]
                idx = arr["indices"]; tr = np.ones(len(s["y"]), dtype=bool); tr[idx] = False
                va = ~tr; assert_split(s, tr, va)
                if kind == "subject":
                    assert set(s["subject"][tr]).isdisjoint(s["subject"][va])
                detailed.append({"model": name, "diagnostic": val["diagnostic"], "accuracy": val["accuracy"],
                                 "macro_f1_all8": val["macro_f1"], "n": val["n"], "classes_present": val["classes_present"]})
            true = np.concatenate([a["true"] for a in arrays]); pred = np.concatenate([a["pred"] for a in arrays])
            row = {"model": name, "diagnostic": kind,
                   "mean_accuracy": float(np.mean([a["accuracy"] for a in values])),
                   "sd_accuracy": float(np.std([a["accuracy"] for a in values], ddof=1)),
                   "pooled": score(true, pred), "folds": values}
            diagnostic.append(row)
    write(REPORT/"split_diagnostics.json", diagnostic)
    write(REPORT/"split_diagnostics_details.json", detailed)
    allfits = [json.loads(p.read_text()) for p in (OUT/"results").glob("**/summary.json")]
    costs = {"completed_fit_calls": len(allfits), "summed_fit_seconds": float(sum(r["fit_seconds"] for r in allfits)),
             "screen_fits": 285, "inner_fits": 345, "adaptive_fits": 110,
             "outer_fits": len(list((OUT/"results/outer").glob("**/summary.json"))),
             "diagnostic_fits": len(list((OUT/"results/diagnostics").glob("**/summary.json"))),
             "failures": len(list((OUT/"results").glob("**/failure.json"))),
             "first_completed_at": min(r["timestamp"] for r in allfits), "last_completed_at": max(r["timestamp"] for r in allfits),
             "execution": "user PC CPU, at most two jobs x two threads", "new_installs": [],
             "A_B_test_evaluations": 0, "GPU_training_in_this_ML_stage": False}
    write(REPORT/"resources.json", costs)

    # Key comparisons show one-factor contrasts as well as all candidates.
    keys = ["legacy_svm1", "fusion_svm1", "fusion_f64_svm1", "fusion_et3",
            "fusion_et3_hierarchy", "fusion_et3_independent", "fusion_lda",
            "fusion_ind_lda", "legacy_ind_lda", "robust_ind_lda", "mixed_et_lda_lowref5s"]
    fig, ax = plt.subplots(figsize=(10, 6.3))
    yy = np.arange(len(keys))
    ax.barh(yy, [byname[n]["mean_accuracy"]*100 for n in keys],
            color=["#c47724" if byname[n]["context"] == "5s" else "#287ca1" for n in keys])
    ax.errorbar([byname[n]["mean_accuracy"]*100 for n in keys], yy,
                xerr=[byname[n]["sd_accuracy"]*100 for n in keys], fmt="none", ecolor="#2f3b48", capsize=3)
    ax.set(yticks=yy, yticklabels=keys, xlabel="A 개발 기록 5겹: 시행 정확도 (%)", xlim=(0, 80))
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(FIG/"development_contrasts.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh(np.arange(len(rows)), [r["mean_macro_f1"] for r in rows],
            color=["#c47724" if r["context"] == "5s" else "#287ca1" for r in rows])
    ax.set(yticks=np.arange(len(rows)), yticklabels=[r["model"] for r in rows], xlabel="개발 5겹 macro-F1 평균")
    ax.tick_params(axis="y", labelsize=7); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG/"all_candidates.png"); plt.close(fig)
    best = selection["best_1s"]
    confusion([pooled["legacy_svm1"]["confusion_matrix"], pooled[best]["confusion_matrix"]],
              ["기존 SVM: 개발 OOF", "독립 LDA: 개발 OOF"], "development_confusion.png")
    confusion([nested[k]["pooled"]["confusion_matrix"] for k in ["baseline", "single", "ensemble"]],
              ["SVM 기준선: 중첩 외부", "23종 선택: 중첩 외부", "확률2종 평균: 중첩 외부"], "nested_confusion.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(diagnostic_lock["models"]):
        value = next(r for r in diagnostic if r["model"] == name and r["diagnostic"] == "subject")
        ax.plot(np.arange(8), [r["accuracy"]*100 for r in value["folds"]], marker="o", label=name)
    ax.set(xticks=np.arange(8), xticklabels=diagnostic_lock["subjects"], xlabel="학습에서 제외한 참가자",
           ylabel="해당 참가자의 개발 시행 정확도 (%)", ylim=(0, 100))
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(FIG/"held_person_diagnostic.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r["mean_train_accuracy"]*100 for r in rows], [r["mean_accuracy"]*100 for r in rows],
               c=["#c47724" if r["context"] == "5s" else "#287ca1" for r in rows], alpha=.6)
    offsets = {"legacy_svm1": (-45, -26), "fusion_et3": (-12, 20), best: (-8, 7)}
    for n in ["legacy_svm1", "fusion_et3", best]:
        q = byname[n]; ax.annotate(n, (q["mean_train_accuracy"]*100, q["mean_accuracy"]*100),
                                  xytext=offsets[n], textcoords="offset points", ha="right", fontsize=8,
                                  arrowprops={"arrowstyle": "-", "color": "#526174", "lw": .7})
    ax.plot([0, 100], [0, 100], "--", color="gray", alpha=.5)
    ax.set(xlabel="학습 정확도 평균 (%)", ylabel="기록 분리 개발 정확도 평균 (%)", xlim=(0, 103), ylim=(0, 103))
    fig.tight_layout(); fig.savefig(FIG/"overfitting_gap.png"); plt.close(fig)

    comparison = {"best_1s": best, "development": rows, "pooled": pooled, "nested": nested,
                  "diagnostics": diagnostic, "resources": costs}
    write(REPORT/"all_comparisons.json", comparison)
    tables = "\n".join(f"| `{v['model']}` | {v['context']} | {100*v['mean_accuracy']:.2f} ± {100*v['sd_accuracy']:.2f} | {v['mean_macro_f1']:.4f} | {100*v['mean_angle_accuracy']:.2f} | {v['five_fold_fit_seconds']:.2f} |" for v in rows)
    nt = "\n".join(f"| {name} | {100*v['mean_accuracy']:.2f} ± {100*v['sd_accuracy']:.2f} | {v['mean_macro_f1']:.4f} |"
                   for name, v in nested.items())
    dt = "\n".join(f"| `{v['model']}` | {v['diagnostic']} | {100*v['mean_accuracy']:.2f} ± {100*v['sd_accuracy']:.2f} | {v['pooled']['macro_f1']:.4f} |"
                   for v in diagnostic)
    b = byname[best]
    doc = f"""# 전통 머신러닝 중심의 팔 동작 분류 탐색

기존 에너지 기준 1초 입력을 중심으로 A 개발 기록 분할에서 79개 후보를 비교했다.
77개 후보는 1초 입력만 사용하고, 2개 후보는 같은 시행의 5초에서 낮은 진폭 기준을 추가했다.
손동작과 각도를 각각 학습한 공분산 축소 선형판별분석(LDA)이 개발 평균 정확도
{100*b['mean_accuracy']:.2f}%, macro-F1 {b['mean_macro_f1']:.4f}로 선택됐다.
기존 집계 SVM의 개발 정확도는 {100*byname['legacy_svm1']['mean_accuracy']:.2f}%였다.
**이 수치는 개발 자료에서 후보를 선택한 결과다. 이번 ML 단계에서는 기존 A/B 시험을 평가하지 않았다.**
논문의 약 96.72% 수준에 도달했다는 근거는 확보하지 못했다.

## 1. 문제와 평가 범위

과제는 8명의 5채널 신호로 4개 손동작과 팔꿈치 30°/60°를 구분하는 8클래스 분류다. 기존 A 개발 자료는
51개 기록, 1,530개 시행이다. 한 기록에 같은 참가자·클래스의 30개 시행이 들어 있다.
기록을 제외하면 그 사람의 해당 클래스 예시가 모두 빠지며, 다른 클래스 기록은 남을 수 있다.
새 피험자 평가와는 다른 조건이다.

원자료·강의 확인 범위는 [데이터 출처](../../DATA_PROVENANCE.md)에 있다.
신호는 이미 필터링되어 있고 마커의 생리적 의미가 불확실해, 기존 신호 에너지로
선택한 1초 구간을 유지했다. 새 ML 입력 캐시에는 A 시험 시행을 제외한 1,530행만 저장했다.

| 평가 | 모델 선택과 표본 단위 | 해석 |
|---|---|---|
| 개발 비교 | 기존 A 기록 5겹, 시행 예측 | 79종 선택에 사용한 개발 성능 |
| 고정 23종 중첩 CV | 외부 5겹·내부 3겹 모두 기록 그룹 | 내부에서 선택한 절차의 외부 결과; 적응적 22종은 제외 |
| 새 기록 분할 진단 | 고정 모델, 새 seed 두 개의 기록 5겹 | 분할 구성 민감도; 추가 선택에 미사용 |
| 참가자 제외 진단 | 고정 모델, a–h를 한 명씩 학습에서 제외 | 참가자 변화 민감도; 후보 선택에 같은 개발 자료를 사용한 이력 존재 |
| 기존 A/B 시험 | 이번 단계에서 미사용 | 이전 시험 결과와 노출 이력 보존 |

중첩 CV에서도 과거 이 데이터로 모델을 설계한 이력까지 사라지지는 않는다.
참가자 제외 진단은 각 fit에서 해당 사람을 제외하지만, 모델 선택까지 독립인
중첩 LOSO 평가는 아니다. 새 독립 시험자료는 현재 없다.

## 2. 관찰에서 가설로

앞선 실험은 손동작보다 각도 구분에서 오류가 많았고, 학습 정확도와 기록 분리 검증의
차이가 컸다. 이를 바탕으로 네 가지 질문을 세웠다. 창의 평균만으로 놓친 분포가 있는지,
채널 사이 관계가 필요한지, 불필요한 특징과 복잡한 경계가 과적합을 키우는지,
각도 학습을 손동작별로 나누거나 전체 손동작에서 모으는 것이 어떤 영향을 주는지다.

사용자가 제시한 가설은 같은 손동작에서 이두근과 삼두근의 반응 방향이 다를 수 있다는
것이다. 이전 개발 21쌍의 관찰은 CH4 증가·CH5 감소 4쌍, 두 채널 증가 16쌍,
CH4 감소·CH5 증가 1쌍이었다. 이번 채널 선택·진폭 비율 대조는 그 관찰과
분류 정보의 관계를 살피는 실험이다. 채널 진폭의 변화 방향은 사람·동작에 따라
달랐으며, 실제 근력이나 이완을 확인하는 측정은 없었다.

## 3. 특징과 모델

| 특징군 | 차원 | 입력과 목적 |
|---|---:|---|
| legacy | 140 | 기존 70특징 × 9창의 평균·표준편차 |
| robust | 280 | 같은 특징에 중앙값·IQR 추가 |
| spectrum | 105 | Welch 대역전력, 평균·중앙 주파수, 엔트로피·평탄도, RMS |
| relations | 150 | 공분산 행렬 로그, 채널·포락선 상관, 상대 RMS·비율, coherence |
| fusion | 395 | legacy+spectrum+relations |
| relative | 320 | 절대 진폭 일부를 제외하고 상대 전력·trace 정규화 공분산 사용 |
| upper / forearm | 56 / 84 | 기존 특징 중 CH4·5 / CH1–3만 사용 |

Welch 전력 추정에는 길이 256표본의 Hann 창과 128표본 간격을 사용했다.
coherence는 교차 스펙트럼의 크기 제곱을 두 신호의 전력 스펙트럼 곱으로
나눈 값이며, 네 주파수 구간의 평균을 사용했다
([SciPy 정의](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html)).
공분산은 전체 및 네 대역 FFT 투영 신호에서 계산하고 5% 대각 축소 후 행렬 로그를
벡터화했다. 이는 시행 내부 특징이며 기록 전체의 검증 통계를 학습에 사용하지 않는다.

SVM은 RBF 커널에서 C=0.1/1/10을 비교했다. RF와 ExtraTrees는 300개 트리,
최대 특징 비율 0.5, 말단 최소 표본 1/3/6개, seed=42를 사용했다.
Histogram GB는 100회 반복·학습률 0.05·L2=1·최대 말단 7/31개로 고정하고 내부 무작위
시행 조기종료를 껐다. LDA는 `solver=lsqr, shrinkage=auto`다.
공분산 축소는 특징 수에 비해 표본이 적을 때 추정 변동을 줄이는 규제다
([LDA 공식 문서](https://scikit-learn.org/stable/modules/lda_qda.html)).
공분산 축소를 적용한 LDA와 비선형 분류기의 결과를 같은 기록 분할에서 비교했다.

표준화, RobustScaler, 누적 설명 분산 95%의 PCA와 특징 선택은 각 학습 폴드에서만 fit했다.
F통계 특징 선택은 학습 기록별 특징 중앙값을 사용해 같은 기록의 반복 시행을
독립 기록처럼 계산하지 않았다. 선택 개수 32/64는 후보 설정으로 사전 고정했다.
SVM 확률 보정용 내부 무작위 CV는 사용하지 않았다.

분류 구조는 8클래스 직접 분류, 4동작·2각도의 독립 분류, 손동작별 2각도 계층
분류를 비교했다. 독립 구조는 각도 학습에 모든 손동작의 표본을 모은다.
계층 구조는 추론할 때 손동작의 정답을 주지 않고, 예측한 동작 확률과 조건부
각도 확률을 곱한다. 모델별 실제 설정은 [고정 설정](artifacts/results/CONFIG_FROZEN.json)에 있다.

## 4. 단계별 판단과 요인별 비교

초기 57종은 특징군·모델군·규제와 구조를 비교했다. 여러 요소가 함께 바뀐 결합 특징
비교는 탐색으로 표시했다. 이어 독립 구조의 이득을 확인하기 위해 22개 대조군을
추가했다. 손동작 모델을 고정하고 각도 입력을 줄이는 대조도 포함했다.
모든 시도의 관찰→가설→변경/고정 조건→결과/변동→해석→결정은
[전체 실험 기록](EXPERIMENT_LOG.md)에 후보별로 정리했다.

| 같은 조건에서 비교한 변화 | 개발 정확도(%) | 해석 범위 |
|---|---:|---|
| fusion ExtraTrees 직접→독립 | {100*byname['fusion_et3']['mean_accuracy']:.2f}→{100*byname['fusion_et3_independent']['mean_accuracy']:.2f} | 분류 구조의 영향 |
| fusion LDA 직접→독립 | {100*byname['fusion_lda']['mean_accuracy']:.2f}→{100*byname['fusion_ind_lda']['mean_accuracy']:.2f} | 같은 특징·LDA에서 구조 비교 |
| 독립 LDA legacy→fusion | {100*byname['legacy_ind_lda']['mean_accuracy']:.2f}→{100*byname['fusion_ind_lda']['mean_accuracy']:.2f} | 특징 결합의 추가 이득은 작음 |
| fusion SVM 전체→F상위64 | {100*byname['fusion_svm1']['mean_accuracy']:.2f}→{100*byname['fusion_f64_svm1']['mean_accuracy']:.2f} | 학습 기록 기준 특징 선택 효과 |
| fusion 독립 ExtraTrees leaf3→6 | {100*byname['fusion_et3_independent']['mean_accuracy']:.2f}→{100*byname['fusion_ind_et6']['mean_accuracy']:.2f} | 트리 복잡도 변화 |

![주요 요인별 비교](artifacts/figures/development_contrasts.png)

*그림 1. 같은 개발 기록 5겹에서의 시행 정확도 평균과 표준편차.
주황색은 5초의 낮은 진폭 기준을 추가한 입력이다. 이 후보는 1초 입력과 문맥 길이가 다르다.*

![과적합 차이](artifacts/figures/overfitting_gap.png)

*그림 2. 각 후보의 평균 학습·개발 정확도. 점선은 두 값이 같은 위치다.
많은 후보가 학습 자료를 거의 완전히 맞히면서 기록 분리 개발 점수는 낮았다.*

## 5. 전체 개발 결과

평균과 표준편차는 같은 5개 기록 폴드의 단순 평균과 표본 표준편차(ddof=1)다.
선택 기준은 8클래스 macro-F1 평균이며 동률은 이름순이다. 학습 시간은 5회 fit의
합계로, 특징 계산·저장·보고서 시간은 제외했다. 한 구조가 여러 분류기를 학습하면
그 시간을 모두 포함한다.

| 후보 | 문맥 | 정확도 평균±SD(%) | macro-F1 | 각도 정확도(%) | fit합계(s) |
|---|---|---:|---:|---:|---:|
{tables}

![79개 후보 전체](artifacts/figures/all_candidates.png)

*그림 3. 79개 후보의 개발 macro-F1. 선택 기준에 따른 정렬이며 성능이 하락한 후보도 포함했다.*

![개발 혼동행렬](artifacts/figures/development_confusion.png)

*그림 4. 동일한 1,530개 개발 시행의 폴드 밖 예측(OOF). 숫자는 건수, 색은 정답 행별 비율이다.
각 예측은 해당 기록을 학습에서 제외한 모델에서 나왔지만, 이 점수들은 후보 선택에 사용했다.*

대표 모델의 손동작 정확도는 {100*b['mean_gesture_accuracy']:.2f}%,
각도 정확도는 {100*b['mean_angle_accuracy']:.2f}%였다. 독립 구조의 개발 이득이
있었지만 각도 구분과 기록별 변동은 남아 있다. CH4·CH5만 사용하거나 진폭을 모두
상대화한 후보도 자동으로 개선되지 않았다.

## 6. 고정 23종의 중첩 평가

외부 폴드별 내부 3겹에서 후보를 선택했다. 단일 선택은 23종 중 macro-F1 최고값,
앙상블은 SVM을 제외한 확률 모델 상위 2개의 동일 가중 평균이다.
모든 선택을 잠근 뒤 외부 결과를 계산했다. 앙상블 가중치를 외부 점수에 맞춰
조정하지 않았다. **이 표는 적응적으로 추가한 독립 LDA 대표의 중첩 성능이 아니다.**

| 절차 | 외부 정확도 평균±SD(%) | 외부 macro-F1 평균 |
|---|---:|---:|
{nt}

![중첩 외부 혼동행렬](artifacts/figures/nested_confusion.png)

*그림 5. 고정 후보 선택 절차의 외부 폴드 예측 합계. baseline은 기존 집계 SVM,
single은 내부 선택 1종, ensemble은 내부 확률 모델 상위 2종의 평균이다.*

## 7. 고정 모델의 분할 민감도

기존 SVM, 초기 대표 ExtraTrees, 전체 1초 대표 LDA, 5초 기준 추가 모델을 고정하고
새 기록 분할 두 개와 참가자 제외 진단을 수행했다. 진단 결과를 본 뒤 대표를 바꾸지 않았다.
`subject`의 평균 정확도는 8명에 동일 가중치를 준다. 각 사람의 기록 수·클래스가
다르므로 아래 F1은 1,530시행 예측을 합친 8클래스 macro-F1이다.

| 고정 모델 | 진단 | 정확도 평균±SD(%) | 합산 macro-F1 |
|---|---|---:|---:|
{dt}

![참가자 제외 진단](artifacts/figures/held_person_diagnostic.png)

*그림 6. 각 fit에서 한 참가자의 개발 기록 전체를 제외했다. 후보 선택에는 같은 개발 자료의
이전 결과를 사용했으므로, 이 그림은 독립적인 신규 참가자 시험 성능을 확정하지 않는다.*

## 8. 이전 결과·논문과의 관계

앞선 [GPU·강의 조건 보고서](../generalization_v2/README.md)의 1초 CNN은 기존
A 시험 50.51%, 새 개발 선택 CNN은 45.64%였다. 이 결과는 그대로 보존하며 이번 ML의
개발 54.65%를 그 시험 점수와 직접 비교하지 않는다. B의 97.14%는 동일 기록 내
마지막 6시행, C의 RF 96.20%는 무작위 시행 5겹의 19창 다수결이다.

[논문 공개 초록](https://link.springer.com/article/10.1007/s13534-026-00607-7)의
RF 96.72±0.58%는 확인했지만 전문의 분할·입력 구성이 아직 미확인이다.
따라서 C의 높은 숫자나 이번 개발 개선을 논문 재현 성공으로 해석할 수 없다.
XGBoost/LightGBM/CatBoost는 확인한 기존 환경에 없어서 이번 무설치 후보에 포함하지 않았다.

## 9. 검증·자원·후속 판단

총 {costs['completed_fit_calls']}개 모델 fit을 완료했다. 모델 내부의 독립·계층 분류기 수는
서로 다르므로 이것은 최상위 후보/폴드 실행 수다. 합산 fit 시간은
{costs['summed_fit_seconds']/60:.2f}분이며 두 작업을 병렬로 실행한 실제 경과시간과 다르다.
CPU는 최대 2작업×2스레드를 사용했다. 이번 전통 ML 단계는 CPU 실행이며,
GPU 사용 근거는 앞선 TITAN Xp CNN 실험에 있다. 새 설치나 외부 연산은 없었다.

특징 해시·그룹 분할·개발행만 포함됐는지 검사했고, 공분산 정규화와 상관의 이득
불변성도 합성 신호로 확인했다. 기존 SVM의 다섯 폴드 예측은 이전 CPU 결과와 완전히
일치했다. 외부 모델은 저장 후 다시 불러와 예측, 학습 통계와 지표를 검사했다.
확률의 병렬 합산 오차는 1e−12 이내로 허용하고 실제 최대 오차를 기록했다.
근거는 [입력 검사](artifacts/reports/initial_verification.json)와
[외부 재현 검사](artifacts/reports/outer_verification.json)에 있다.

이번 탐색에서는 전통 ML의 분류 구조를 바꿨을 때 개발 성능이 개선됐지만,
논문에 보고된 수치에는 미달했고 평가 조건도 달랐다. 특징·정규화·모델군·복잡도·
채널·분류 구조를 대조한 뒤에도 각도 오류와 분할 변동이 남았다. 같은 개발 기록에서
더 큰 격자를 반복하면 후보 선택 편향이 커지므로, 이번 결과를 고정하고 다음 정보가
필요한 단계로 정리했다. 논문 전문은 정확한 재현 조건을, 독립 세션과 전극 위치·실제
관절각·동작 시점 기록은 각도와 측정 조건의 영향을 구분하는 데 필요하다.
이번 ML 단계에서는 기존 A/B 시험에 대한 추가 튜닝·재평가를 하지 않았다.

## 10. 재현과 파일 구조

```text
experiments/ml_v3/
  PROTOCOL.md, AMENDMENT_1.md, DIAGNOSTICS_PLAN.md
  prepare.py, models.py, train.py, adaptive.py, lock.py
  diagnostics.py, verify.py, report.py
  README.md, EXPERIMENT_LOG.md
  artifacts/  # 설정·집계·예측·로그·그림; 원신호·특징행렬·모델 제외
```

새 작업 루트의 준비와 CPU 기준선 생성 순서는 [루트 재현 절차](../../README.md#144-개발-전용-전통-ml의-전체-재현)에 있다.

원래 저장소의 데이터 준비와 앞선 `generalization_v2/prepare.py`를 완료한 작업 루트에서
실행한다. 작업 루트 바로 아래에 저장소와 `step2_gpu_pc`, `step2_improvement_v2`,
`step2_ml_v3`가 나란히 놓인다. 새 결과를 재생성할 때에는 비어 있는 별도 루트를 사용한다.
실행과 구문 검사는 conda Python 3.13.9에서 확인했다. 시스템 Python 3.10에서는
특징 준비 코드의 확장 인덱싱 문법이 지원되지 않으므로 아래의 conda 실행기를 사용한다.
라이브러리 버전은 [환경 기록](artifacts/reports/environment.json)에 있다.

```bash
mkdir -p ../step2_ml_v3/{{data,results,reports,logs}}
PY=/home/ckh/anaconda3/bin/python
$PY experiments/ml_v3/prepare.py
$PY experiments/ml_v3/train.py self-check
# partition0/1은 각 후보를 나누며 동시에 실행할 때 합계4thread
$PY experiments/ml_v3/train.py screen 0
$PY experiments/ml_v3/train.py screen 1
$PY experiments/ml_v3/train.py inner 0
$PY experiments/ml_v3/train.py inner 1
$PY experiments/ml_v3/adaptive.py prepare
$PY experiments/ml_v3/adaptive.py 0
$PY experiments/ml_v3/adaptive.py 1
$PY experiments/ml_v3/lock.py
$PY experiments/ml_v3/diagnostics.py prepare
$PY experiments/ml_v3/verify.py initial
$PY experiments/ml_v3/train.py outer 0
$PY experiments/ml_v3/train.py outer 1
$PY experiments/ml_v3/diagnostics.py 0
$PY experiments/ml_v3/diagnostics.py 1
$PY experiments/ml_v3/verify.py outer
$PY experiments/ml_v3/report.py
```

설정과 선택 파일은 덮어쓰지 않는다. 원신호·특징 캐시·학습 모델은 로컬
`step2_ml_v3`에 있고 검토용 결과 묶음에서 제외했다. 로그 사본의 개인 홈·작업 루트는
치환했으며 원본 로그는 보존했다. 초기 공개 결과는 기존 커밋에 보존했고, 누적 결과는 루트 README에 통합했다. 후속 [XGBoost 비교](../xgb_v4/README.md)는 별도 환경과 계획으로 수행했다.
"""
    (HERE/"README.md").write_text(doc)
    artifact = HERE/"artifacts"
    shutil.copy2(DATA/"manifest.json", REPORT/"input_manifest.json")
    for part in ["results", "reports", "figures", "logs"]:
        for p in (OUT/part).rglob("*"):
            if p.is_file() and p.suffix in [".json", ".jsonl", ".npz", ".csv", ".png", ".log"]:
                dst = artifact/p.relative_to(OUT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if p.suffix == ".log":
                    dst.write_text(p.read_text().replace(str(TASK), "<WORKSPACE>").replace("/home/ckh", "<USER_HOME>"))
                else:
                    shutil.copy2(p, dst)
    print(json.dumps({"best": best, "development_accuracy": b["mean_accuracy"],
                      "nested": {k: v["mean_accuracy"] for k, v in nested.items()},
                      "diagnostics": [{k: v[k] for k in ["model", "diagnostic", "mean_accuracy"]} for v in diagnostic],
                      "resources": costs}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
