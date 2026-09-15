# Week 2 - 멀티암드 밴딧 (Multi-Armed Bandit)

## bandit.py
- `Bandit`: 팔(arm)마다 고정된 승률(rate)을 갖는 정상(stationary) 밴딧 환경.
- `Agent`: 표본 평균(sample average) 방식으로 가치(Q)를 갱신하는 ε-greedy 에이전트.

## non_stationary.py
- `NonStatBandit`: 매 스텝마다 각 팔의 승률이 랜덤하게 흔들리는 비정상(non-stationary) 밴딧 환경.
- `AlphaAgent`: 고정된 학습률 α로 Q값을 갱신하는 ε-greedy 에이전트 (최근 보상에 더 큰 가중치).

## parameter_a.py
- 비정상 환경에서 표본 평균 에이전트(`Agent`)와 다양한 α값(`AlphaAgent`, α=0.2/0.5/0.8)을 비교.
- 200회 반복 × 1000 스텝 동안 평균 보상률을 계산해 그래프로 비교.

## parameter_epsilon.py
- 정상 환경(`Bandit`)에서 ε값(0.1, 0.3, 0.01)을 바꿔가며 성능을 비교.
- 200회 반복 × 1000 스텝 동안 평균 보상률을 계산해 그래프로 비교.

## graph_a.png / graph_e.png
- 각각 `parameter_a.py`, `parameter_epsilon.py` 실행 결과 그래프.
- 그래프 분석 및 최종 분석은 [analysis.md](./analysis.md) 참고.
