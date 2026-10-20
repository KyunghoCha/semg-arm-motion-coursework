"""Create a reviewable local report without modifying the published main README."""
from pathlib import Path
import json, shutil
from common import *


def main():
    r = json.loads((OUT/"reports/all_comparisons.json").read_text())
    lock = json.loads((OUT/"results/ADAPTIVE_SELECTION_LOCK.json").read_text())
    selected = r["selected"]; final = r["final"][selected]
    artifact = HERE/"artifacts"
    artifact.mkdir(exist_ok=True)
    for part in ["reports", "figures", "results", "logs"]:
        for p in (OUT/part).rglob("*"):
            if p.is_file() and p.suffix in [".json", ".jsonl", ".csv", ".png", ".npz", ".log"]:
                destination = artifact/p.relative_to(OUT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if p.suffix == ".log":
                    destination.write_text(p.read_text().replace(str(TASK), "<WORKSPACE>")
                                           .replace("/home/ckh", "<USER_HOME>"))
                else:
                    shutil.copy2(p, destination)
    # Feature arrays and trained weights are intentionally outside this package.
    drows = "\n".join(f"| `{v['model']}` | {100*v['mean_accuracy']:.2f} ± {100*v['sd_accuracy']:.2f} | {v['mean_macro_f1']:.4f} | {v['epochs'] if v['model'] not in CPU else '—'} |"
                      for v in r["development"])
    frows = "\n".join(f"| `{name}` | {100*v['accuracy']:.2f} | {v['macro_precision']:.4f} | {v['macro_recall']:.4f} | {v['macro_f1']:.4f} | {100*v['gesture_accuracy']:.2f} | {100*v['angle_accuracy']:.2f} |"
                      for name, v in r["final"].items())
    crows = "\n".join(f"| {v['assembly']} | {v['model']} | {v['unit']} | {100*v['cv_mean_accuracy']:.2f} ± {100*v['cv_sd_accuracy']:.2f} | {v['cv_macro_f1']:.4f} | {100*v['holdout_accuracy']:.2f} |"
                      for v in r["classroom"])
    nrows = "\n".join(f"| {'기존 CNN' if name == 'baseline' else '16종 내부 선택'} | {100*v['mean_accuracy']:.2f} ± {100*v['sd_accuracy']:.2f} | {v['mean_macro_f1']:.4f} |"
                      for name, v in r["nested"].items())
    note = f"""# 기록 일반화 추가 실험: GPU 모델·특징 비교

이 보고서는 같은 1초 입력에서 기록 간 일반화를 개선할 수 있는지 검토한 로컬 실험을 정리한다.
개발 검증에서는 대표 CNN의 평균 시행 정확도가 37.88%에서 48.85%로 높아졌지만,
재사용 A 시험에서는 50.51%에서 45.64%로 낮아졌다. **새 모델의 기록 일반화 개선은 확인되지 않았다.**
강의 조건을 별도로 구현한 C의 RF는 시행 다수결에서 5겹 평균96.20%였다.
이 수치는 같은 기록 안의 시행 분할 결과이며 A와 평가 대상이 다르다.

기존 공개 결과와 코드는 커밋 `264f38a38246308235b30fbe77a0be7bb21f54cd`에 보존했다.
이번 코드는 로컬 `research/generalization-v2` 브랜치에 있으며 원신호·강의 원본·모델은
이 검토용 결과 묶음에 포함하지 않았다.

## 1. 데이터와 비교 기준

8명, 5채널, 4손동작×30°/60°의8클래스다. 64기록은 각각30시행을 포함한다.
클래스 순서는 주먹30/60, 펴기30/60, 회내30/60, 회외30/60이다.
원출처와 강의4개113쪽의 확인 범위는 [데이터 출처](../../DATA_PROVENANCE.md)에 있다.

| 조건 | 개발·학습과 평가 대상 | 이번 사용 |
|---|---|---|
| A | 51기록1,530시행 개발 / 13기록390시행 시험 | 개발5겹, 중첩5×3기록CV, 재사용 시험 |
| B | 기록별 첫24시행 개발 / 마지막6시행 시험 | 기존97.14%를 보존; 이번에 재학습·재평가하지 않음 |
| C | 전체1,920시행의 무작위 층화5겹과 별도80/20 | 강의 조건을 명시한 SVM/RF 기준 실험 |

기록은 한 참가자·한 클래스의30시행으로 구성된다. A에서 기록을 제외하면 그 참가자의
해당 클래스 예시가 모두 학습에서 빠진다. 같은 사람의 다른 클래스는 학습에 남을 수 있다.
각 개발 폴드의 클래스별 학습 기록은4–6개다. 따라서1,530시행을1,530개의 독립 측정 조건으로
해석하기 어렵다. A는 새 피험자 평가가 아니다.

모든 겹치는 창은 시행 안에 남긴다. 표준화·특징 선택과 조기종료는 학습/개발 폴드에서
결정했다. 초기16종의 중첩 평가에서는 각 외부 폴드의 내부3겹으로 모델과 epoch를 고르고,
다섯 선택을 모두 고정한 뒤 외부 점수를 계산했다. 보충7종은 적응적 개발 비교로 구분한다.
이미 이 데이터의 검증·시험 결과를 여러 차례 확인했으므로 중첩 구성도 과거 설계 노출을
제거한 신규 독립 시험을 의미하지 않는다.

## 2. 강의·논문과 구현의 대응

강의는2초에서200ms창을100ms간격으로 자르고, RMS·MAV·WL·ZC·SSC의25특징을
설명한다. 1.5–3.5초는 강의 예제의 시작점 가정이며 논문에서 확정된 구간이 아니다.
제공 신호는 이미 필터링돼 있어 이번에도 추가 필터링하지 않았다. 원자료의 마커와
생리적 동작 시작의 대응은 불명확해 A는 기존 에너지 최대1초를 그대로 사용했다.

[공식 초록](https://link.springer.com/article/10.1007/s13534-026-00607-7)은
RF96.72±0.58%와5겹CV를 보고한다. 전문이 없어 분할 단위, 구간 시작, 창 집계와
표준화의 정확한 순서는 확인하지 못했다. C는 이 불확실성을 별도 설정으로 명시한
강의 기준 실험이다. 창별 학습 후 다수결과 특징을 시행별 평균한 학습을 구분했다.
전체 수치와 조건은 [사전 계획](PROTOCOL.md)에 있다.

## 3. 관찰에서 후보 선택까지

| 관찰 | 가설·변경 | 고정 조건 | 개발 결과와 판단 |
|---|---|---|---|
| 이전CNN의 각도 혼동 | 손동작4개·각도2개 보조 손실 | 기존1초·64표본STFT·기록5겹 | 46.52%; 후속 비교에 유지 |
| 마커 의미 불명확 | 에너지2초, 전체5초 RMS 포락선 | 같은 기록 그룹과 학습 통계 범위 | 2초37.05%; 포락선29.24–32.48%; 미채택 |
| 진폭의 기록 차이 | 포락선 상대값·결합, 상대 스펙트럼 CPU | 표본 안에서만 정규화 | 일관된 향상 없음 |
| 보조 손실과 클래스 가중치가 함께 바뀜 | 보조 손실 없는 가중치 대조군 | 60epoch상한·조기종료·seed 유지 | 38.21%; 보조 과제 패키지의 개발 이득 유지 |
| 짧은 STFT의 주파수 해상도 | Hann128/hop16, Hann256/hop32 | 같은 원파형1초 | 43.04%,47.38%;256을 후속 비교 |
| 채널별 진폭 변화 | 학습 시 채널·공통 진폭 증강 | 검증에는 증강 없음 | 64STFT46.06%;256조합48.85% |
| 같은 동작의 각도 오류 | 손동작별 각도 확률과RMS보조 입력 | 실제 손동작 정답은 추론에 미제공 | 44.28%,45.07%; 대표 모델보다 낮음 |
| 단일 모델 변동 | 개발상위2종의 동일 가중치 앙상블 | 가중치 튜닝 없음 | 47.50%,F1 .4216; 선택 모델 F1보다 낮아 기각 |

이 표의 가설은 결과를 검토한 실험 설계자의 판단이다. 사용자의 이두근·삼두근 가설은
기존 문서의 범위를 유지했다. 개발21쌍을 다시 계산한 결과 CH4증가·CH5감소4쌍,
두 채널 증가16쌍, CH4감소·CH5증가1쌍이었다. 이 관찰은 모든 사람에게 같은 각도별
진폭 규칙을 적용하기 어려움을 보여준다. 전기 신호 진폭만으로 근력이나 실제 이완을
확정할 수 없고, 각도와 별도 기록의 측정 조건도 분리되지 않는다.

![개발 기록의 CH4·CH5 분포](artifacts/figures/development_channel_distribution.png)

*그림1. 같은 에너지1초에서 각 기록의30시행 RMS 중앙값을 표시했다. 원은30°, 삼각형은60°이며
같은 색은 같은 참가자다. 두 각도가 모두 개발 데이터에 있는 참가자·동작은 선으로 연결했다.
축은 로그 눈금이다. 기록별 절대 진폭의 분포와 각도 간 변화 방향을 함께 확인한다.*

## 4. 모델과 학습 설정

스펙트럼 CNN은 채널×주파수를 입력 채널로 사용한1차원 합성곱3층, 평균·최대 집계,
64차원 은닉층과8클래스 출력을 사용한다. 보조 과제는 손동작4개와 각도2개의 추가
교차엔트로피에 각각0.3을 곱한다. 대표 모델은75,982파라미터다.

AdamW(lr.001, weight decay.001), batch128, dropout.25, label smoothing.05,
gradient clip5를 사용했다. 기존 CNN은 최대40/min8/patience8, 새 모델은
최대60/min12/patience12였다. 새 모델의 클래스 가중치는 학습 클래스 빈도로 계산했다.
진폭 증강은 채널별·공통 자연로그 이득을 각각[-.15,.15] 균등분포에서 뽑는다.
STFT256은645채널×24시간 프레임이며 Hann창과 무패딩을 사용한다.
모든 신경망은 연결된 PC의 TITAN Xp에서 학습했다.

CPU는 기존70특징의9창 평균·표준편차, 상대 대역전력·상관,2초25특징과5초포락선을
SVM 또는 shrinkage LDA로 비교했다. SVM은 학습 표준화와 RBF gamma=scale을 썼다.
후속 전통ML 중심의 더 넓은 비교는 별도 `../ml_v3`에 기록한다.

## 5. 전체 개발 비교

아래 정확도는5개 기록 폴드의 단순 평균±표준편차(ddof1), F1은8클래스 macro-F1의
폴드 평균이다. 시행 수가330인 폴드와300인 폴드가 있어 합산 시행 정확도와 차이가 난다.
epoch는 폴드별 최적epoch의 중앙값이며 최종 재학습에 사용했다.

| 후보 | 시행 정확도(%) | macro-F1 | 최종epoch |
|---|---:|---:|---:|
{drows}

![전체 개발 비교](artifacts/figures/development_models.png)

*그림2. 23종의 기록 분리 개발 정확도와 폴드 표준편차. 순서는 선택 기준인 macro-F1 기준이다.
주황색이 개발에서 선택한 대표 모델이다. 입력1초·2초·5초 후보는 코드명과 사전 계획에서 구분한다.*

![개발 학습 곡선](artifacts/figures/learning_curves.png)

*그림3. 기존 모델과 선택 모델의5개 개발 폴드 학습 곡선. 점선은 macro-F1로 고른epoch다.
학습 정확도가 계속 증가하는 동안 기록 분리 검증 정확도는 낮게 머물러 과적합을 보였다.
서로 다른 목적함수를 쓰므로 학습 손실 크기 자체를 모델 간 성능 차이로 해석하지 않았다.*

## 6. 중첩 기록 평가

초기16종만 포함하는 고정 선택 절차의 결과다. 보충7종과 최종 대표 모델의 중첩 점수로
해석할 수 없다. 모델·epoch 선택은 내부3겹에서 끝내고 외부 폴드에서는 고정epoch로 학습했다.

| 평가 절차 | 시행 정확도 평균±SD(%) | macro-F1 평균 |
|---|---:|---:|
{nrows}

![중첩 폴드 비교](artifacts/figures/nested_folds.png)

*그림4. 같은 외부 기록 폴드에서 기존CNN과 내부 후보 선택 절차를 비교했다.
선택 절차의 평균 정확도는 기존CNN보다 낮았으며 개발 화면의 높은 최고값이 그대로 유지되지 않았다.*

![중첩 혼동행렬](artifacts/figures/nested_confusion.png)

*그림5. 외부 폴드 예측을 합친1,530시행 혼동행렬. 숫자는 건수, 색은 정답 행별 비율이다.*

## 7. 재사용 A 시험: 한 번의 고정 평가

모든 후보·epoch·앙상블 여부를 고정한 뒤 각 신경망을 seed42와2026으로 재학습했다.
아래는 두 seed 확률을 동일하게 평균한 결과다. 개별 seed 결과도 보존했다.
개별 seed 중 점수가 높은 실행을 사후 대표값으로 선택하지 않았다.

| 모델 | 정확도(%) | macro정밀도 | macro재현율 | macro-F1 | 손동작(%) | 각도(%) |
|---|---:|---:|---:|---:|---:|---:|
{frows}

기존 CNN의197/390개 정답과 확률은 보존된 결과와 일치했다. 선택 모델은178/390개를
맞혀 기존보다4.87%p 낮았다. 참가자6개를 군집으로 재표집한 기술적95%구간은 차이에 대해
−17.78~+9.52%p였다. 표본 수와 반복 사용 이력이 있어 이를 확증적 유의성 판단으로 사용하지 않았다.

![A 시험 혼동행렬](artifacts/figures/final_confusion.png)

*그림6. 같은390시행에서의 비교. 선택 모델의 오분류{final['errors']}건 중
{final['same_gesture_angle_errors']}건이 동일 손동작의 반대 각도였다.
손동작 정확도{100*final['gesture_accuracy']:.2f}%와 각도 정확도{100*final['angle_accuracy']:.2f}%의 차이가 남았다.*

## 8. 강의 기준 C 결과

C는1.5–3.5초에서19창을 추출했다. `window_vote`는 창을 학습하고19예측을 다수결한다.
`trial_mean`은19특징벡터를 평균해 시행 하나를 학습하는 입력 구성 민감도 비교다.
다수결 동률은 가장 작은 클래스ID로 정했다. 두 구성 모두 같은 시행의 창은 분리하지 않았다.
무작위 분할seed42를 고정했으며 좋은 seed를 골라 보고하지 않았다.

| 입력 구성 | 모델 | 단위 | 5겹 정확도 평균±SD(%) | macro-F1 평균 | 80/20평가 정확도(%) |
|---|---|---|---:|---:|---:|
{crows}

RF100트리·seed42, SVM RBF C1·gamma=scale을 사용했고 표준화는 학습에서만 fit했다.
RF 창 정확도와 시행 다수결의 차이는 집계 효과다. 같은 기록이 양쪽에 포함되므로
높은 시행 점수는 새 기록/새 사람 일반화 근거가 아니다. XGBoost는 기존 설치에 없어
실행하지 않았다. C의 결과는 논문 목표 수치에 가깝지만 정확한 논문 재현은 확인되지 않았다.

![강의 조건 C 혼동행렬](artifacts/figures/classroom_confusion.png)

*그림7. C의80/20평가384시행. 두 모델 모두 창별 학습 후19창 다수결로 집계했다.*

실행별4지표·학습 시간·100회 단일 입력 추론 시간은
[C 원결과](artifacts/results/classroom/all_results.json)에 있다.
입력이 창인지 시행 요약인지에 따라 추론1회의 의미가 다르다.
동일5폴드의 RF–SVM 대응 t검정도 [집계JSON](artifacts/reports/all_comparisons.json)에 기록했으며,
학습집합의 중첩·동일 참가자 반복과 다중 비교를 고려하지 않은 기술적 비교로 한정했다.

## 9. 실행·재현 검사와 남은 원인

총396개 모델 fit을 완료했다(개발115, 내부240, 외부10, A최종7, C24).
그중 CUDA fit191회의 합산 학습 시간은{r['resources']['gpu_summed_fit_seconds']/60:.2f}분이며
특징 생성·검사·문서 작업 시간은 제외한다. GPU 최고온도79°C, PyTorch 최대할당269.17MiB,
장치 전체 최대사용1,169MiB였다. Python3.10/PyTorch2.6.0+cu118,
CUDA런타임11.8, 드라이버580.178.04를 사용했다. `nvidia-smi`의CUDA13.0 표시는
드라이버 호환 한계이며 학습 런타임 버전과 구분한다. CPU는 기존 conda Python3.13환경이다.
새 설치·클라우드 연산·원격 게시·LMS제출은 하지 않았다.
검토용 로그 사본에서는 사용자 홈과 작업 루트 경로를 치환했고 원본 로그는 로컬에 보존했다.

원자료64개 해시, 특징 캐시 해시, 고정 코드 해시, 모든 분할·예측 인덱스를 검사했다.
신규 모델 경로는 합성자료 학습과 저장 상태의 예측 일치 검사를 통과했다.
최종6개 CNN 체크포인트를 다시 불러와 확률의 완전 일치와 학습행만으로 계산한
정규화 통계의 일치를 확인했다. 근거는 [검사JSON](artifacts/reports/verification.json)과
[체크포인트 검사](artifacts/reports/final_checkpoint_verification.json)에 있다.

이번 범위에서 남은 핵심 문제는 각도 구분과 기록별 분포 차이다. 같은 기록 안에서는
높은 정확도가 나왔지만 기록을 바꾸면 학습 곡선과 외부 평가가 크게 약해졌다.
전극 위치 변화, 세션 순서, 실제 관절각과 동기 동작 시점은 이 자료만으로 분리할 수 없다.
이 때문에 단일 원인을 확정하지 않았다. 논문 전문과 독립 세션/참가자 자료는
비교 조건 확정과 새로운 일반화 평가에 필요하다.

이 GPU 단계의 평가 후에는 후보를 수정하지 않았다. 이후 사용자가 요청한 전통ML 중심
개발 탐색은 `ml_v3`의 별도 사전 계획과 개발 전용 입력으로 진행하며 기존 A/B시험을 추가 사용하지 않는다.

## 10. 재현 실행

새 작업 루트의 전체 준비와 기존 CNN 예측 복사는 [루트 재현 절차](../../README.md#143-추가-cnncpu강의-기준-c의-전체-재현)를 따른다.

현재 PC에서는 저장소 루트에서 아래 순서로 실행한다. 기존 결과 경로가 있으면 완료한
학습을 건너뛰며, 선택 파일을 다시 만드는 명령은 기존 파일을 보호하도록 실패한다.
완전 재실행에는 비어 있는 별도 작업 루트를 사용한다.

```bash
# 원자료와 과거 입력 준비: run.py의 별도 workspace 기능 사용
/home/ckh/anaconda3/bin/python run.py download --workspace ../step2_gpu_pc
/home/ckh/anaconda3/bin/python run.py features --workspace ../step2_gpu_pc
/home/ckh/anaconda3/bin/python run.py prepare --workspace ../step2_gpu_pc

# 새 작업 루트에서만 결과 폴더를 준비
mkdir -p ../step2_improvement_v2/{{data,results,reports,logs}}
/home/ckh/anaconda3/bin/python experiments/generalization_v2/prepare.py
/usr/bin/python3 experiments/generalization_v2/gpu.py self-check
/home/ckh/anaconda3/bin/python experiments/generalization_v2/cpu.py screen
/home/ckh/anaconda3/bin/python experiments/generalization_v2/cpu.py nested_inner
/usr/bin/python3 experiments/generalization_v2/gpu.py screen
/usr/bin/python3 experiments/generalization_v2/gpu.py nested_inner
/usr/bin/python3 experiments/generalization_v2/common.py
/home/ckh/anaconda3/bin/python experiments/generalization_v2/classroom.py
/usr/bin/python3 experiments/generalization_v2/phase2.py prepare
/usr/bin/python3 experiments/generalization_v2/phase2.py self-check
/usr/bin/python3 experiments/generalization_v2/phase2.py screen
/usr/bin/python3 experiments/generalization_v2/phase3.py prepare
/usr/bin/python3 experiments/generalization_v2/phase3.py self-check
/usr/bin/python3 experiments/generalization_v2/phase3.py screen
/usr/bin/python3 experiments/generalization_v2/evaluate.py lock
/usr/bin/python3 experiments/generalization_v2/gpu.py nested_outer
/home/ckh/anaconda3/bin/python experiments/generalization_v2/cpu.py final
/usr/bin/python3 experiments/generalization_v2/evaluate.py final
/usr/bin/python3 experiments/generalization_v2/evaluate.py verify
/home/ckh/anaconda3/bin/python experiments/generalization_v2/report.py
/usr/bin/python3 experiments/generalization_v2/package_report.py
```

작업 루트 바로 아래에 저장소 폴더, `step2_gpu_pc`, `step2_improvement_v2`가 나란히
있어야 한다. 공개 재현 코드와 같은 과거 helper를 원 작업 폴더에서 import한다.
실행 코드·설정·환경을 임의로 바꾸면 고정 해시 검사가 중단한다. 모델 가중치는
로컬 `step2_improvement_v2/results`에 보존돼 있다.
"""
    (HERE/"README.md").write_text(note)
    (OUT/"REPORT.md").write_text(note.replace("artifacts/", "")
        .replace("../../DATA_PROVENANCE.md", "../semg-arm-motion-coursework/DATA_PROVENANCE.md")
        .replace("(PROTOCOL.md)", "(../semg-arm-motion-coursework/experiments/generalization_v2/PROTOCOL.md)"))
    print("packaged", artifact, "README", HERE/"README.md")


if __name__ == "__main__":
    main()
