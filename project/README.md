# 두 팔 로봇 운반 · PPO · A2C · SAC

MuJoCo·Gymnasium에서 두 팔 로봇이 1kg 물체를 잡아 운반하도록 학습합니다. PyTorch 기반 PPO·A2C·SAC가 **각각 독립 환경 1개**를 사용합니다. 관측 105개·연속 행동 14개이며 모델·옵티마이저·정규화·승급·기록을 공유하지 않습니다.

## 설치와 실행

Conda 환경만 사용합니다.

```bash
cd /home/a/Reinforcement_Learning_class/project
conda activate dual-reacher
python -m pip install -r requirements-study.txt
./reconnect_web.sh
./start_three_models.sh
```

각 모델은 **단계마다 최대 1,000만 스텝**을 학습합니다. CPU 물리·CPU 신경망이 기본이며 화면은 웹에서 봅니다. **검증 성공률 80% 이상을 두 번 연속** 달성하면 같은 단계의 다음 레벨로, 레벨 4에서는 다음 단계의 레벨 0으로 넘어갑니다. 다음 단계에 들어갈 때 그 단계의 카운터가 0으로 돌아가며, 모델과 전체 누적 기록은 이어집니다. 80% 미만이면 연속 횟수가 초기화됩니다. 승급은 모델별로 독립입니다. 한 단계의 1,000만 스텝을 사용하고도 그 단계를 통과하지 못하거나, 4단계 레벨 4를 통과하면 종료합니다.

2단계는 그 실행에서 완료한 1단계 정책으로 실제 잡고 든 상태를 준비합니다. 학습하는 모델과 옵티마이저는 계속 이어가며, 그 이전 실험의 체크포인트 경로를 입력할 필요는 없습니다.

재부팅 후에는 같은 폴더에서 다음처럼 재개합니다.

```bash
conda activate dual-reacher
./reconnect_web.sh
./start_three_models.sh --resume
```

완료된 모델은 건너뛰고 남은 모델만 재개합니다. 모델·옵티마이저·관측 정규화와 SAC 리플레이 버퍼를 복원하지만, 진행 중이던 물리 에피소드는 다음 초기화 시드로 새로 시작합니다. 난수 궤적까지 같은 재개는 보장하지 않습니다. 중단은 실행 터미널에서 Ctrl+C입니다.

## 웹에서 보기

[기존 웹 주소](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/)에서 PPO·A2C·SAC를 선택합니다. 본인 계정 전용이며 다른 Wi-Fi에서도 접속할 수 있습니다.

팔 마스크 시험은 `./start_masked_study.sh`로 실행합니다. 모델당 **65,536스텝**, 1단계 난도 0에 고정됩니다. 자동 승급을 사용하는 장시간 학습은 `./start_masked_study.sh --full`입니다. 기존 방식과 같이 **단계당 최대 1,000만 스텝**을 사용합니다. PPO·A2C는 평소 2,048스텝을 수집하며 단계 한도에 닿는 마지막 수집은 남은 스텝만 진행합니다. 세 모델은 같은 고정 관측 통계를 사용하고, SAC는 경험 재사용 방식을 유지합니다.

로컬에 모델별 실시간 창이 열립니다. 짧은 시험은 `runs/studies/three_models_1kg_masked/`에 저장되며 최종 양팔 허용 결과는 각 모델의 `final_unmasked_validation.json`에서 볼 수 있습니다. 장시간 학습은 별도의 `runs/studies/three_models_1kg_masked_full/`에서 처음 시작하며 웹 실험 목록의 **한 팔 마스크 · 장시간 학습**에서 확인합니다. 중단한 장시간 실행을 이어갈 때는 `./start_masked_study.sh --full --resume`을 사용합니다. 웹 화면만 사용하려면 `--no-local-viewer`를 붙입니다.

학습 중 로컬에서도 세 모델의 영상 창을 보려면 활성화된 Conda 환경에서 `./show_training_windows.sh`를 실행합니다. 현재 학습의 실시간 영상에 연결하며, 영상 창을 닫아도 학습은 계속됩니다.

별도 안정성 비교 실험은 `python -m evaluation.stability_experiment --output runtime/experiments/stability_trial_1`로 실행합니다. PPO·A2C의 최고 모델 복사본에서 기준 설정·정규화 고정·수집량 확대를 조건별 131,072스텝씩 순차 비교합니다. CPU 한 개를 낮은 우선순위로 사용하며 결과 CSV·JSON·PNG·SVG만 남깁니다. 현재 학습이나 웹에는 자동 적용하지 않습니다.

- 모델별 실제 시뮬레이션 영상, 누적 스텝, 현재 난이도, 성공률.
- 에피소드 총보상·성공·길이·힘 사용량과 모델 업데이트 표.
- 학습 중 약 1분마다 갱신하는 모델별 그래프와 종료 후 최종 자료.
- 모델별 최고 검증 정책의 평가 50회 중 상위 10개 영상.

순위 카드의 `평가 에피소드 35`는 보상순 정렬 전 35번째 평가라는 뜻입니다. 학습의 35번째 에피소드는 아닙니다. 각 모델의 최고 도달 난도와 검증 성능으로 정책을 고르므로 서로 다른 난도의 보상을 직접 비교하지 않습니다. 첫 평가 영상은 초기 검증·생성이 끝나면 나타납니다.

1단계의 최고 정책은 성공률 → 안정된 들기율 → 실제 잡기율 → 평균 총보상 순으로 고릅니다. 아직 성공하지 못했어도 물체를 잡는 정책이 가만히 있는 정책보다 우선하며, 선택된 정책의 평가 에피소드 상위 10개는 총보상으로 정렬합니다.

## 저장 위치

기본 실행은 `runs/studies/three_models_1kg_v3/{ppo,a2c,sac}/`에 분리 저장합니다. 첫 실험은 `models/first_trial/`에 보존했고, 중단 이유는 그 안의 `README.md`, 모델별 최종 그래프는 `graph/{ppo,a2c,sac}/`에 있습니다. 현재 실행은 타이머 관측과 보상을 일치시킨 v4 보상을 사용하며 첫 실험 모델을 이어받지 않습니다. 사전 시험에서 유지 타이머의 정규화 포화를 보완하고 PPO 업데이트 폭을 줄였습니다. 이 설정은 이전 실행의 체크포인트와 구분합니다.

| 경로 | 내용 |
|---|---|
| `latest/`, `best/` | 재개용·최고 검증 모델과 정규화, 필요한 SAC 버퍼 |
| `episodes.csv`, `metrics.csv`, `validation_*.json` | 에피소드·업데이트·검증 원본 |
| `mission_replays/` | 상위 10개 영상·시작/최대 높이/종료 장면·보상 시계열 |
| `recordings/` | 500만 스텝 단위의 50fps·1배속 평가 영상과 PNG |
| `reports/`, `reports/stage1/` | 300dpi PNG·SVG·원본 CSV·발표 메모 |
| `artifacts/` | 시작 설정·라이브러리 버전·소스와 해시 |

모델은 Git이나 웹에 올리지 않습니다. 비공개 웹 연결 설정은 `runtime/`과 로그인 키링에 유지합니다. 모델·로그의 저장과 별개로 채팅에는 반복 학습 로그를 보내지 않습니다.

새 보상은 [reward.md](reward.md), 교육과정 검토는 [curriculum_review.md](curriculum_review.md), 추가 연구는 [additional_study.md](additional_study.md)에 있습니다. 기존 native PPO 실행기는 호환용으로 남아 있고 기존 보상 설명은 [reward_current.md](reward_current.md)입니다.
