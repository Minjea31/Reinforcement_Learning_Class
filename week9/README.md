# 강화학습 week9 — Quiz Q1

이론·과제 근거는 `08-Neural Networks.pdf`만 사용했다. 공식 DeZero는 vendor 폴더의 실행 의존성이다. 최신 보고서는 `~/ppt_workspace/output/RL_week9_Q1/RL_week9_Q1_v4.pptx`와 PDF다.

## 실행

```bash
cd ~/Reinforcement_Learning_class/week9
.env/bin/python train_final.py --seed 0  # 최종 모델 새 학습·결과 출력
.env/bin/python controlled_experiment.py # 파라미터 한 조건씩 변경 + 조합 재검증
.env/bin/python plot_controlled.py       # 조건별 정답/예측 함수 비교
.env/bin/python visualize_v4.py          # 최종·기준 모델 시각화
.env/bin/python analyze_results_v4.py    # 구간·극값·시드 분석
.env/bin/python check_results.py         # 조건 통제·모델 수치·자동미분 검증
```

이미 완료한 조건은 설정 해시별 캐시를 재사용한다. 새로 실행하려면 `results`를 다른 곳에 옮기고 먼저 `experiment.py`(1차 탐색), 다음 `controlled_experiment.py`를 실행한다. 실제 추가 실험은 CPU 프로세스 3개에서 수행한다.

`.env`는 시스템 패키지를 읽는 별도 가상환경이다. 다른 PC에서는 `python3 -m venv .env` 후 `.env/bin/python -m pip install -r requirements.txt`로 환경을 구성한다. DeZero는 `vendor/deep-learning-from-scratch-3/dezero`에서 직접 import하며 원본 라이선스와 버전은 vendor·environment.json에 보존했다. PyPI 연결이 불가해 공식 소스를 받아 사용했으며 자동미분과 전체 학습을 검증했다.

## 데이터와 평가

- 학습점 100개, `x=rand(100,1)`, `y=sin(6πx)+randn(100,1)`. 잡음은 사용자 선택으로 강의 52쪽의 표준편차 1을 따른다.
- 시드 0·1·2에서 입력·잡음·초기값이 함께 달라진다. 같은 시드의 비교 조건은 동일 데이터를 사용한다.
- 검증점: `i/400 (i=0,...,400)`. 평가점: `i/1001 (i=1,...,1000)`. 둘은 겹치지 않으며 학습에 사용하지 않는다.
- 모든 후보는 DeZero MSE와 역전파로 실제 학습한다. 모델 선택은 3시드 검증 MSE 평균을 사용한다. 최종 성능은 별도 평가 지점에서 측정한다.
- 그래프의 조건별 회귀 함수는 seed 0의 학습 모델이다. 오차 수치는 세 시드 평균이므로 한 곡선으로 전체 평균을 대표하지 않는다.
- PDF/PPT 표시만 소수점 3자리로 반올림했다. 계산 및 원본 JSON·CSV는 정밀도를 유지한다. 작은 설정값은 과학적 표기로 표시한다.

## 비교 조건

1차 탐색(`experiment.py`)으로 기준 후보를 선정한 뒤, 기준 후보에서 한 번에 한 파라미터만 바꾸어 재검증했다.

기준 후보: 1층·50노드·tanh·Xavier·Adam(0.01)·weight decay 0.0001·9,900회·전체 배치 100개.

|항목|실제 변경 후보|
|---|---|
|노드 수|10, 50, 100|
|은닉층 수|1, 2, 3 (각 50노드)|
|optimizer|SGD, MomentumSGD, AdaGrad, Adam (학습률 0.01 고정)|
|activation function|sigmoid, tanh, ReLU (출력층은 선형)|
|weight initialization|표준편차 1, 0.01, Xavier, He (bias=0)|
|learning rate|0.0001, 0.001, 0.01, 0.1|
|weight decay|0, 0.0001, 0.001, 0.01|
|updates|1,000, 3,000, 6,000, 9,900|

그룹별 최저 조건을 합친 10노드·He·학습률 0.001 조합도 실제로 학습했다. 조합 검증 MSE는 약 0.208로, 단일 초기화 변경 후보의 0.128보다 높았다. 실제 학습 후보 중 검증 오차가 가장 낮은 모델을 최종 채택했다. 따라서 각 단일 비교의 최저값을 무조건 합치지 않았다.

## 현재 최종 모델

- 1층·50노드·tanh·He·선형 출력. 파라미터 151개.
- Adam 학습률 0.01, β1=0.9·β2=0.999·ε=1e-8. 전체 배치 100, weight decay=0.0001, 9,900회 업데이트.
- He σ=√(2/fan-in), 모든 층에 같은 규칙, bias=0. dtype=float64.
- 목적함수: 표본 평균 MSE + 0.5 λ ΣW². bias에는 L2를 적용하지 않는다. 보고서 학습 MSE에는 L2 항이 포함되지 않는다.
- 최종 검증 MSE 평균 0.1282542537546292.
- 최종 독립 평가 MSE 평균 0.1274256567710237. 시드별 0.12844377002701696, 0.04560745829017794, 0.20822574199587615.
- 기준 평가 평균 0.5159115273369507 대비 약 75.301% 감소. 시드별 편차와 극값·경계 오차가 남아 있다.
- Momentum=0.9, AdaGrad ε=1e-8. 이들은 라이브러리 기본값이다. 비유한 손실 또는 학습 MSE>1e12는 발산으로 처리한다.

## 기록

- `results/controlled_summary.json`: 그룹별 변경 조건, 검증 MSE와 편차, 각 시드 기록, 조합 모델, 최종 선택.
- `results/summary_v4.json`: 최신 보고서 기준·최종 모델과 평가 결과.
- `results/detailed_analysis_v4.json`: 시드별 MSE·MAE·최대 절대오차, 주기별 MSE, 정답 함수 극값 위치의 예측값.
- `results/controlled_conditions.csv`: 변경 조건별 결과 목록.
- `results/figures_v4/`: 최신 보고서 그림과 조건별 학습 함수.
- `results/<설정해시>_<시드>.json/.npz`: 실제 실행 로그와 모델 가중치. `_best.npz`는 참고용이며 이번 고정 조건 실험의 선택에는 사용하지 않았다.
- 기존 `summary.json`, `detailed_analysis.json`, `figures/`는 이전 설계 기록이다. 현재 결과는 반드시 v4 파일과 구분한다.

PPT 제작·렌더링·검수 코드는 `~/ppt_workspace/ppt_info/RL_week9_Q1`에 있다. PDF는 PPTX 도형·텍스트·이미지를 읽는 로컬 렌더러로 생성했다. PowerPoint 자체 엔진의 실제 표시는 검증하지 못했다.
