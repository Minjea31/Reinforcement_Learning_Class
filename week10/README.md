# 강화학습 week10 — Quiz Q1 / Q2

강의 `09-Deep Q-Network.pdf` p.10·13·32–35·41 기반. Python 3.14, NumPy 2.4.6, Gymnasium 1.2.3에서 실행했다. DeZero 공식 소스는 vendor에 라이선스와 함께 포함한다. PPT 제작 의존성은 필요 없다.

```bash
python -m venv .env
.env/bin/pip install -r requirements.txt
.env/bin/python experiment.py q1 --workers 8
.env/bin/python evaluate.py q1 --seed 0
.env/bin/python check_results.py
# Q1 결과와 PPT 정리를 마친 다음 Q2 실행
.env/bin/python experiment.py q2 --workers 8
.env/bin/python evaluate.py q2 --seed 0
.env/bin/python check_results.py
```

- `experiment.py`: 신경망·GridWorld·DQN·실험 행렬·모델 선택. `evaluate.py`: 저장 모델의 greedy 정책 재평가. `check_results.py`: 환경 규칙·체크포인트·단일 변경 통제 검사.
- `results/q1|q2/summary.json`: 기준·후보·모든 시드 결과·조합 검증·최종 선택. 각 설정 해시 폴더에 시드별 JSON·NPZ 저장. 완료한 실험은 캐시 재사용. 완전 재실험은 results를 다른 폴더로 옮긴 뒤 실행.
- OFAT: 모든 후보가 동일 기준에서 한 조건만 다르다. 시드 0·1·2. 신경망 구조의 층수·너비는 독립 항목이다. 비교 후보는 experiment.py의 BASE·SWEEPS에 명시.
- Q1: 0 기준 좌표. 시작 (4,0), 목표 (0,4), 벌점 (0,3),(3,4), 벽 (2,1),(2,2). 목표만 종료하며 벌점은 통과 가능. 최대 200스텝은 bootstrap을 차단하지 않는 인위적 학습 제한. 22개 비종료·이동 가능 상태에서 greedy 평가. 같은 gamma의 가치 반복 해는 검증용이며 학습에 사용하지 않음.
- Q1 선택: 도달률 → 동률 허용 최적 행동 일치율 → 평균 경로 길이 → Q RMSE의 사전식 순서. 서로 다른 gamma의 할인 반환값을 직접 순위에 사용하지 않는다. Q 테이블의 벽·종료 상태 출력은 의미 없음.
- Q2: CartPole-v1, 최대 500스텝. Adam·경험 재생·타깃망. terminated만 bootstrap 차단, truncated는 유지. 강의 요약표 기준 epsilon=.05, buffer_size=100000 (p.32 코드의 .1,10000과 다름). 타깃 동기화는 에피소드 단위.
- 검증 초기 상태 seed=500000+학습seed*100+i, i=0..19. 검증 평균 보상으로 최종 설정 선택, 동률이면 시드 간 표준편차 작은 설정. 독립 최종 평가 seed=900000+학습seed*100+i, i=0..99. 학습은 학습seed*100000+episode.
- 배치 표본은 중복 없이 추출. 에피소드 수 비교는 같은 시드·나머지 설정에서 학습 앞부분이 일치한다. 버퍼 크기 비교에서 실제 경험량이 작은 경우 큰 버퍼 후보들이 동일하게 작동할 수 있다.
- 각 항목의 우수값 조합도 실제 재학습한다. 최종 모델은 실행한 단일 변경 후보·기준·조합 전체 중 선택한다. 전역 최적이나 통계적 유의성을 주장하지 않는다.
- PPT 제작 코드는 `/home/a/ppt_workspace/ppt_info/RL_week10_full`에 별도 보관. 최종 PPTX·PDF는 `/home/a/ppt_workspace/output/RL_week10_full`.

## 추가 검증·내보내기

`python test_training.py`: 재생 버퍼 준비 중 종료/reset 처리와 타깃망 기울기 분리 회귀 검사. `python export_results.py`: 전체 설정별 CSV와 Q2 독립 평가 1회 상태 궤적 생성. `check_results.py`는 에피소드 수 비교의 동일 학습 구간 일치와 최종 저장 모델의 검증 점수 재현도 검사한다.

실행 결과: Q1 18설정×3시드=54회, Q2 27개 단일 비교 설정+조합 1개=28설정×3시드=84회. 최종 Q1은 4000에피소드 설정, Q2는 할인율 0.9 설정. Q2 독립 300회 평균 398.22점으로 완전 해결이나 모든 초기 상태의 성공을 주장하지 않는다.
