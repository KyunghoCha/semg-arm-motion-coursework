# 전통 ML 중심 추가 탐색: 사전 계획

이 단계는 사용자가 전통 머신러닝 중심의 폭넓은 탐색과 과정 기록을 요청한 뒤 시작한다.
직전 GPU 비교와 그 시험 결과는 보존한다. 현재 공개 main은 변경하지 않는다.
코드는 `experiments/ml_v3`, 비공개 입력과 실행 결과는 형제 폴더
`step2_ml_v3`에 둔다. 패키지를 설치하거나 외부 서비스를 사용하지 않는다.

## 문제 관찰과 범위

A 개발 기록의 이전 1초 CNN은 손동작보다 각도 구분이 어려웠다. 시간 구간을
늘리거나 전체 진폭을 제거하는 변경도 일관된 개선을 보이지 않았다. 시행마다
5채널의 주파수 분포와 채널 관계, 시간적 변동을 요약하고, 여러 결정 경계와
손동작·각도 분류 구조를 비교한다. 이는 실험 설계자의 가설이다.
사용자의 이두근·삼두근 반응 방향 가설은 기존 문서의 실제 발언과 관찰 범위를 유지한다.

새 실험의 데이터는 **기존 A 개발 51기록·1,530시행으로 제한**한다.
기존 A/B 시험은 모델 선택과 새 평가에 사용하지 않는다. 이전에 같은 데이터의
검증과 시험을 여러 번 분석했으므로 이번 검증도 독립적인 신규 데이터 평가가 아니다.
새 분할 이름으로 이 노출 이력을 초기화하지 않는다.

## 입력과 특징 가설

모든 후보는 기존 에너지로 선택한 동일한 1초를 사용한다. 파형의 추가 필터링과
구간 최적화는 하지 않는다. 한 시행 안에서 계산하는 특징만 미리 캐시한다.

| 특징군 | 구성 | 검토할 가설 |
|---|---|---|
| legacy | 기존 70특징의 9창 평균·표준편차,140차원 | 기존 CPU 기준 재현 |
| robust | 같은70특징의 평균·표준편차·중앙값·IQR,280차원 | 일시적인 큰 진폭과 분포 비대칭 |
| spectrum | 256표본 Welch의 8대역 상대·절대 전력, 평균·중앙 주파수, 엔트로피, 평탄도, RMS | 주파수 분포와 진폭의 분리 |
| relations | 5대역 공분산의 행렬 로그, 파형·100ms RMS 포락선 상관, 채널 로그RMS 비율, coherence | 개별 채널 특징에서 빠진 관계 |
| fusion | legacy+spectrum+relations | 상보적 정보 결합; 여러 요인이 바뀌는 탐색 |
| relative | legacy의 절대 진폭15열 제외, 상대 주파수, trace정규화 공분산, 채널 관계 | 전체 진폭 변화 민감도 감소 |
| upper / forearm | legacy에서 CH4·5 또는 CH1–3만 사용 | 채널 제거의 영향; 근력·이완의 인과 검정은 아님 |

Welch/coherence는 Hann256, hop128, detrend constant이다. 대역 경계는
20,50,80,110,150,200,260,330,451Hz다. coherence는 20–80/80–150/
150–250/250–451Hz 평균이다. 공분산은 전체 파형과 같은4개 대역의 FFT
투영 파형에서 계산한다. 이는 특징 계산용 투영이며 새로운 전처리 필터로
원자료를 덮어쓰지 않는다. 공분산은95%표본공분산+5%trace/5단위행렬,
행렬 로그는 고유값분해로 계산하고 상삼각15개를 벡터화한다.

## 비교와 선택

- 개발 비교: 기존 A의 기록 그룹5겹을 유지한다. 시행/겹치는 창/기록이 넘어가지 않는다.
- 표준화, robust scaling, PCA, 특징 선택은 각 학습 폴드에서만 fit한다.
- 특징 선택의 F통계는 **학습 기록별 특징 중앙값**에서 구한다. 같은 기록의
  30시행을 독립 기록30개처럼 특징 선택에 세지 않는다.
- 일차 선택 기준은5겹 macro-F1 평균이며 동률은 후보 이름 순이다.
  정확도, 표준편차, 손동작/각도 정확도, 학습 시간과 모든 실패를 함께 기록한다.
- RF/ExtraTrees는300트리, seed42, 최대2thread. Histogram GB는
  반복 수를 사전에 고정하고 내부 무작위 시행 조기종료를 끈다.
- SVM은 RBF, probability=False다. SVM 점수를 확률로 보정하는 내부 무작위
  CV를 사용하지 않는다. 주 분류 결과는 estimator.predict를 사용한다.
- 손동작/각도 독립 구조는 두 확률의 곱, 계층 구조는4손동작 확률과
  손동작별2각도 확률의 곱으로8클래스를 예측한다. 실제 손동작 라벨은
  추론 입력으로 제공하지 않는다. 구조 비교는 ExtraTrees로 한다.

전체 후보와 파라미터는 첫 학습 전에 `CONFIG_FROZEN.json`에 보존한다.
일차 중첩 검증군은6특징군×(SVM C1, ExtraTrees leaf3, shrinkage LDA)의18개,
legacy SVM C10, fusion RF leaf3, fusion Histogram GB, fusion의 독립/계층
ExtraTrees까지23개다. 기존5외부×3내부 기록 그룹을 그대로 사용한다.
각 외부 폴드의 후보를 내부 결과로만 고르고 모든 선택을 잠근 뒤 외부를 평가한다.
원래 A 개발 전체를 보고 고른 후보 하나의 점수를 중첩 점수로 보고하지 않는다.

추가 개발 비교는 SVM C0.1/C10, RF/ExtraTrees 말단 크기, robust scaling,
PCA95%, 기록 기준 F상위32/64개, 상완/전완 채널 제거와 분리 구조다.
초기 전체 비교 후 최대 한 차례, 유망 후보의 요인을 분리한 보충 비교를
추가할 수 있다. 그 경우 실행 전 변경 계획을 기록하고 일차 중첩군에 섞지 않는다.
동일 가중치 확률 앙상블은 일차 중첩군의 확률 모델 상위2개를 각 내부
검증에서 선택하는 규칙으로 별도 평가한다. SVM은 보정 확률이 없어 제외한다.
앙상블 가중치는 최적화하지 않는다.

## 자원과 중단

기존 conda Python3.13/sklearn1.7.2를 사용한다. SVM·RF·ExtraTrees·Histogram
GB는 이 설치에서 CPU로 실행하며, GPU를 사용했다고 표현하지 않는다.
GPU CNN 결과는 앞선 PC 실행의 비교 자료다. XGBoost/LightGBM/CatBoost는
확인한 기존 환경에 없어 이번 무설치 후보군에는 포함하지 않는다.

최대4CPU thread, 동시에2개 작업, 실행 예산60분, 여유디스크5GiB 이상.
비수치·분할 불일치·해시 불일치 시 중단한다. 모든 성공/실패를 후보별로 기록한다.
외부 평가 후 해당 점수에 맞춘 추가 후보나 파라미터 변경은 하지 않는다.

## 근거와 해석 범위

- 강의의 SVM/RF 및 기존 데이터 구조는 앞선 `generalization_v2/PROTOCOL.md`와
  저장소 `DATA_PROVENANCE.md`의 원문 대조 기록을 따른다.
- ExtraTrees의 무작위 분할과 파라미터:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- Histogram GB의 조기종료/검증 분할:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- 특징 선택의 학습 pipeline:
  https://scikit-learn.org/stable/modules/feature_selection.html
- coherence의 Welch 기반 정의:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html

온라인 문서는 현재 버전이며 실행은 설치된1.7.2/1.16.3의 API를 확인한다.
공분산·주파수 특징의 이 과제 성능은 미확인 가설이며 논문 최적 설정으로 제시하지 않는다.
논문 전문의 분할 단위·입력 구성은 여전히 미확인이다.
