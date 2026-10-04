# Week 4: GridWorld 정책 평가

`gridworld.py`와 `policy_eval.py`는 제공된 Practice #1 스크린샷의
코드 구조, 변수명, 기본값, 계산 순서를 그대로 옮겼습니다.
줄바꿈과 일부 주석의 배치만 정리했습니다.

스크린샷에 구현이 없는 `common/gridworld_render.py`는 별도로 작성한
시각화 보조 모듈입니다. `week4Q1.py` 역시 추가 작성한 실행 파일입니다.
첫 번째 화재 탐지 논문 스크린샷에는 구현 코드가 없어 포함하지 않았습니다.

```bash
python -m pip install -r requirements.txt
python policy_eval.py
python week4Q1.py
```

`policy_eval.py`는 균등 무작위 정책의 가치를 표시합니다.
`week4Q1.py`는 퀴즈의 다섯 정책을 차례로 출력하고 시각화합니다.
각 그래프 창을 닫으면 다음 정책으로 넘어갑니다.
`week4Q1.py`는 항상 그래프를 표시합니다.
그래프는 표시 전에 `week4Q1_graphs/policy_1.png`부터
`policy_5.png`까지 저장됩니다. 재실행하면 같은 파일을 덮어씁니다.

행동 순서는 `[UP, DOWN, LEFT, RIGHT]`, 할인율은 `0.9`입니다.
목표 상태의 가치는 0이고, 목표 칸에 들어갈 때 보상 1을 받습니다.
보상 -1인 칸은 종료 상태가 아니므로 계속 머무르면 보상이 반복됩니다.
강의의 `eval_onestep`은 같은 딕셔너리를 순서대로 갱신하므로,
한 번의 순회에서 이미 갱신한 값을 사용합니다.
강의와 동일하게 `states()`에는 벽 좌표도 포함되지만 그래프와 퀴즈 표에서는
벽의 가치를 표시하지 않습니다.

## Q2: 5×5 GridWorld

```bash
python week4Q2.py
```

`value_iter.py`는 Practice #3의 강의 코드이며 import 경로만 현재 구조에 맞췄습니다.
`gridworld_q2.py`는 Q2 그림의 환경이고, `week4Q2.py`는 단계별 저장을 처리합니다.
폭탄은 Q1과 동일하게 진입 보상 -1을 주는 비종료 칸으로 해석했습니다.

정책 반복은 정책 평가가 끝난 각 바깥 반복마다 저장하고,
가치 반복은 초기 상태 및 각 전체 상태 갱신 결과를 저장합니다.
각 그래프는 저장 후 표시되며 창을 닫으면 다음 단계로 진행합니다.
`week4Q2_graphs/policy_iteration`과 `week4Q2_graphs/value_iteration`에
`step_XX.png`, `step_XX.txt`, 최종 `final.png`, `final.txt`를 저장합니다.
분석 문서는 `week4Q2_graphs/week4Q2_analysis.md`에 자동 생성됩니다.
재실행하면 같은 이름의 결과를 덮어씁니다.
