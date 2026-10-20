"""Audit and report the adaptive head-round comparison, without new fitting."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/semg_xgb_mpl")
import json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from angle_rounds import *
sys.path.insert(0, str(HERE))
from report import aggregate


def main():
    frozen()
    s = dict(np.load(DATA / "splits.npz"))
    timestamp = json.loads((OUT / "results/ANGLE_ROUNDS_FROZEN.json").read_text())["timestamp"]
    checked, max_gesture_difference = 0, 0.
    for name, cfg in CASES.items():
        for k in range(5):
            folder = OUT / f"angle_rounds/{name}/fold{k}"
            row = json.loads((folder / "summary.json").read_text())
            a = np.load(folder / "predictions.npz")
            tr, va = s["fold"] != k, s["fold"] == k
            assert_split(s, tr, va)
            assert timestamp < row["timestamp"] and row["config"] == cfg
            assert np.array_equal(a["indices"], np.flatnonzero(va)) and np.array_equal(a["true"], s["y"][va])
            assert np.array_equal(a["pred"], a["probabilities"].argmax(1))
            assert np.isclose(row["accuracy"], np.mean(a["true"] == a["pred"]))
            assert np.isclose(row["macro_f1"], f1_score(a["true"], a["pred"], labels=range(8), average="macro", zero_division=0))
            base = np.load(OUT / f"results/screen/{cfg['feature']}_independent_d2/fold{k}/predictions.npz")
            # Only angle rounds changed: gesture marginal probabilities must replay.
            error = float(np.max(abs(a["probabilities"].reshape(-1,4,2).sum(2)-base["probabilities"].reshape(-1,4,2).sum(2))))
            assert error < 2e-6, (name, k, error)
            max_gesture_difference=max(max_gesture_difference,error); checked+=1
    assert checked == 80
    candidates = {name: aggregate(sorted((OUT / f"angle_rounds/{name}").glob("fold*/summary.json"))) for name in CASES}
    for feature in ("legacy", "robust", "fusion", "relative"):
        candidates[f"{feature}_angle200"] = aggregate(sorted((OUT / f"results/screen/{feature}_independent_d2").glob("fold*/summary.json")))
    ranking=sorted(candidates,key=lambda name:(-candidates[name]["mean_macro_f1"],name)); best=ranking[0]
    fits=[v for name,r in candidates.items() if not name.endswith('angle200') for v in r['folds']]
    result=dict(timestamp=stamp(),best_development=best,candidates=candidates,
                selection_criterion="same five development recording folds mean macro-F1; ties by name",
                development_only=True,prior_outer_results_seen=True,new_outer_or_test_evaluations=0,
                fit_count=len(fits),fit_seconds=sum(v['fit_seconds'] for v in fits),
                max_sampled_temperature_c=max(v['temperature_c'] for row in fits for v in row['resources']),
                verification=dict(evaluations_recomputed=checked,gesture_probability_max_abs_difference=max_gesture_difference,
                                  only_angle_rounds_changed=True,recording_and_trial_disjoint=True))
    write(OUT / "reports/angle_rounds_comparison.json",result)
    fig,ax=plt.subplots(figsize=(9,5))
    for feature,label in [('legacy','기존 요약'),('robust','분포 요약'),('fusion','결합'),('relative','상대량')]:
        a=[candidates[f'{feature}_angle{v}']['mean_accuracy']*100 for v in (10,25,50,100,200)]
        ax.plot([10,25,50,100,200],a,'o-',label=label)
    ax.set(xlabel='각도 헤드 부스팅 반복 수 (손동작 200 고정)',ylabel='개발 기록 5폴드 시행 정확도 (%)',ylim=(0,75),xticks=[10,25,50,100,200])
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'figures/angle_rounds.png');plt.close(fig)
    rows=['| 특징 | 각도 반복 수 | 시행 정확도 평균±SD (%) | macro-F1 | 각도 정확도 (%) | 학습 정확도 (%) |',
          '|---|---:|---:|---:|---:|---:|']
    for feature in ('legacy','robust','fusion','relative'):
        for rounds in (10,25,50,100,200):
            v=candidates[f'{feature}_angle{rounds}']
            rows.append(f"| {feature} | {rounds} | {v['mean_accuracy']*100:.2f}±{v['sd_accuracy']*100:.2f} | {v['mean_macro_f1']:.4f} | {v['mean_angle_accuracy']*100:.2f} | {v['mean_train_accuracy']*100:.2f} |")
    text='''# 각도 학습량 후속 비교

## 관찰·가설·검증

고정 24종 완료 뒤 개발 곡선에서 각도 검증 손실의 상승을 확인했다. 손동작 학습량은 200으로 유지하고 각도만 10·25·50·100으로 줄였다. 깊이 2·4특징군·나머지 파라미터·기록 5폴드는 동일하다. 각 특징의 200라운드 행은 기존 결과를 재사용했다. 추가 비교 전 중첩 결과도 이미 본 상태이며, 이번 결과는 개발 선택만 수행했다. 기존 시험·중첩 외부·참가자 진단은 반복하지 않았다. [추가 전 고정 계획](ANGLE_ROUNDS_PLAN.md)에 범위를 명시했다.

## 결과

'''+ '\n'.join(rows)+f'''

추가 비교와 기존 기준에서 macro-F1으로 선택된 후보는 `{best}`다. 정확도 {candidates[best]['mean_accuracy']*100:.2f}±{candidates[best]['sd_accuracy']*100:.2f}%, F1 {candidates[best]['mean_macro_f1']:.4f}였다. 이 선택 점수는 외부 성능 추정치가 아니다.

![각도 학습량 대조](artifacts/figures/angle_rounds.png)

*그림 1. 같은 개발 5폴드의 시행 정확도 평균. 모든 점에서 손동작 반복 수는 200이며 각도 반복 수만 다르다. 후보 선택에는 정확도가 아닌 macro-F1을 사용했다.*

## 비용·해석·다음 선택

80회 추가 학습의 합산 fit 시간은 {result['fit_seconds']:.2f}초였다. 학습 손실 평가를 기록하지 않아 앞선 504회의 fit 시간과 계산 범위가 다르다. 저장 예측에서 80회 정확도와 F1을 다시 계산했고, 기존 200라운드 모델과 손동작 주변확률의 최대 차이는 {max_gesture_difference:.3g}로 부동소수점 반올림 범위였다. 바꾼 요인이 각도 학습량임을 대조했다.

학습량 감소는 같은 개발 자료에서 과적합 원인을 비교한 대조다. 독립 세션·신규 참가자의 성능은 이 단계에서 측정하지 않았다. 좁은 반복 수를 다시 탐색하거나 외부 결과로 대표를 변경하지 않았다. 전체 정밀 수치와 폴드 변동은 [집계·검사](artifacts/reports/angle_rounds_comparison.json)에 있다. 재현은 루트 README 14.5의 고정 24종 실행을 마친 뒤 `angle_rounds.py`, `angle_report.py` 순서로 수행한다.
'''
    (HERE/'ANGLE_ROUNDS_RESULTS.md').write_text(text)
    public=HERE/'artifacts'
    for src in (OUT/'angle_rounds').rglob('*'):
        if src.is_file():
            assert src.suffix in ('.json','.npz')
            dest=public/src.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for relative in ('results/ANGLE_ROUNDS_FROZEN.json','reports/angle_rounds_comparison.json','figures/angle_rounds.png','logs/angle_rounds.log'):
        if not (OUT/relative).exists() and relative.startswith('logs/'): continue
        dest=public/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/relative,dest)
    print(json.dumps(dict(best=best,accuracy=candidates[best]['mean_accuracy'],macro_f1=candidates[best]['mean_macro_f1'],fit_seconds=result['fit_seconds'],verification=result['verification']),indent=2))


if __name__=='__main__':main()
