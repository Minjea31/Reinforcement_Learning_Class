# week11 Quiz1 — OpenAI Gym / 강의 1-step Actor-Critic

학습 시드는 0으로 고정한다. OpenAI Gym 0.26.2의 MountainCar-v0를 직접 사용한다.

## 실행

Python 3.10에서 requirements.txt를 설치한 뒤 아래 순서로 실행한다. 영상 인코딩에는 FFmpeg가 필요하다.

```bash
python experiment.py --workers 4
python evaluate.py --video
python check_results.py
```

완료 결과는 설정별 캐시를 재사용한다. 완전 재실험은 results와 combination_retrain을 함께 다른 곳으로 옮긴 다음 실행한다. 프로토콜 해시에 구현 구분을 포함해 이전 확장 실험의 캐시와 구분한다.

## 강의 방식과 구현

- `backend.py`: 공식 OpenAI Gym과 DeZero 소스를 가져오고 `gym.make('MountainCar-v0')`로 실제 환경을 생성한다. 자체 물리 환경은 사용하지 않는다.
- `experiment.py`: ReLU Actor·Critic, Categorical 행동 표본 추출, 환경 1스텝 직후 1-step TD 갱신. `target=reward+gamma*V(next_state)*(1-done)`, target·delta unchain, Critic MSE와 Actor −logπ×delta, 별도 Adam을 사용한다.
- Gym 0.26 API의 `terminated or truncated`를 강의의 done으로 사용한다. 목표 도달과 200스텝 시간 제한에서 bootstrap을 차단한다. 보상은 Gym 원래 −1.000이다.
- 속도 입력 스케일은 MountainCar 관측의 수치 크기를 조절하는 별도 비교 항목이다. 원래 상태에서 선형 배율만 적용한다.
- `evaluate.py`: 저장 DeZero 가중치의 NumPy 추론. 학습·검증·독립 평가·영상에 모두 확률적 행동 선택을 사용한다. 모델 선택은 검증 평균 보상이며 독립 평가 점수로 재선택하지 않는다.
- 영상은 독립 평가 첫 에피소드를 실제 Gym 기본 렌더러로 실행해 기록한다. 초기·종료 화면만 잠시 정지한다. 화면 아래에 스텝·원래 보상·위치·속도를 표시한다.
- `check_results.py`: 실제 Gym 클래스, 매 전이당 갱신 횟수, 원래 보상, 실제 TD 기울기, 저장 모델 확률, 전체 검증 보상, 모델 선택과 독립 평가를 검증한다.

## 파일

results/summary.json에 전체 설정과 선택 결과, 해시별 폴더에 actor.npz·critic.npz·result.json을 저장한다. independent_evaluation.json은 최종 평가 100개의 보상과 궤적이다. comparison.csv는 표시용 비교표이다. 영상과 생성 정보는 videos/에 있다.

## 제공 소스와 글꼴

vendor/gym은 공식 openai/gym의 0.26.2 태그 소스, vendor/deep-learning-from-scratch-3는 강의에서 사용하는 DeZero 소스이다. 각각의 라이선스를 포함했다. 설치된 Gym으로도 같은 버전에서 실행할 수 있다. 영상의 Pretendard Bold 글꼴과 라이선스는 assets/fonts에 포함한다.

## 완료한 실험

기본 8개 항목과 추가 5개 항목을 각각 한 조건씩 변경하여 29개 고유 설정, 조합 반복을 포함한 30회 학습을 완료했다. 선택 설정은 할인율 0.900, 은닉 노드 256, 속도 입력 스케일 10000.000이며 나머지는 기본 탐색 기준값이다. 검증 평균 −118.067, 확률 정책 독립 평가 100회 평균 −120.370, 표준편차 3.580, 도달률 100.000%이다. 영상은 독립 평가 첫 에피소드의 OpenAI Gym 기본 렌더러 실행 화면이다.
