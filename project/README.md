# 두 팔 로봇 운반 · PyTorch PPO

MuJoCo와 Gymnasium으로 만든 두 팔 로봇 환경입니다. 구면 관절 2개와 집게가 있는 팔 두 개로 **5kg 물체**를 잡아 목표에 내려놓습니다. 최종 환경에서는 시작점·목표점과 장애물이 무작위로 바뀌며, 혼자 운반하기·협동하기·넘겨주기는 정책이 학습합니다.

## 웹에서 보기 · Sites

Sites로 배포한 페이지에서 학습 화면과 평가 영상을 볼 수 있습니다.

바로 접속: [https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/)

| 페이지 | 내용 |
|---|---|
| [실시간 학습 화면](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/) | 실제 학습 환경의 연속 영상과 현재 단계·스텝·보상 |
| [현재 정책 상위 10개](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/replays.html) | 영상 생성 시점의 모델을 50회 평가해 보상이 높은 10개를 저장한 50fps 영상 |
| [이전 정책 상위 10개](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/previous-replays.html) | 이전 단일 과제 모델의 평가 영상 |

본인 ChatGPT 계정으로 로그인하는 비공개 페이지이며, 다른 Wi-Fi에서도 접속할 수 있습니다. 실시간 갱신에는 학습 PC와 웹 전송·터널 프로세스가 실행 중이어야 합니다. 저장된 평가 영상은 학습 PC가 꺼져 있어도 볼 수 있으며, 영상에 표시된 모델 스텝과 생성 시각을 기준으로 확인합니다.

## 설치

가상환경은 Conda를 사용합니다. 현재 PC에는 `dual-reacher` 환경을 설치했습니다.

```bash
conda env create -f environment.yml  # 다른 PC에서 처음 설치할 때
conda activate dual-reacher
```

실행 스크립트는 활성화한 Conda 환경의 Python을 사용합니다. PyTorch 2.5.1 CUDA 12.4와 MuJoCo·Gymnasium·Tk를 포함합니다. 물리 계산은 CPU에서, 정책 학습은 `--device cuda`로 GPU에서 실행합니다.

## 현재 모델을 이어서 학습하기

기존 학습이 실행 중이면 해당 터미널에서 `Ctrl+C`로 저장 후 종료하고 실행합니다.

```bash
conda activate dual-reacher
systemd-inhibit --what=sleep --mode=block --why="PPO training" \
  ./run_web_training.sh --resume runs/stage1_100m/latest.pt --device cuda --viewer
```

로컬 창과 Sites 실시간 화면을 함께 연결합니다. 웹 인증 정보는 별도 준비 프로세스의 메모리에서 전달하며 명령어·파일에 저장하지 않습니다. 웹 연결 준비 프로세스와 터널은 재부팅하면 다시 준비해야 합니다. 관련 로그는 `runs/stage1_100m/live_access.log`와 `stream_tunnel.log`에 남습니다.

로컬 화면만 필요하면 같은 옵션으로 `./run_curriculum.sh`를 실행합니다. 단계별 2천만 스텝 일정과 500만 간격 녹화는 체크포인트에서 복원됩니다.

## 4단계 학습 시작

```bash
./run_curriculum.sh --total-steps 100000000 --device cuda --run-dir runs/curriculum_100m --viewer
```

**하나의 모델을 초기화하지 않고 1→2→3→4단계로 이어서 학습**합니다. 관측값 105개와 행동값 14개, 옵티마이저·정규화 상태를 모든 단계에서 유지합니다. 행동은 관절과 집게 목표의 작은 증분이며, 단계 정보도 관측에 포함합니다. 물리 계산은 CPU, 신경망 계산은 GPU에서 실행합니다.

| 단계 | 학습 내용 | 힘·두 번째 팔 비용 |
|---|---|---|
| 1 | 장애물 없는 좌우 대칭 기본 배치에서 잡고 들어 유지하기 | 없음 |
| 2 | 가까운 목표로 옮기고 집게를 열어 내려놓기 | 없음 |
| 3 | 쉬운 배치에서 시작해 랜덤 시작점·목표점·장애물 배치 비율 확대 | 없음 |
| 4 | 전체 랜덤 환경에서 적은 팔과 최소 힘으로 성공 유지하기 | 0→100%로 강화 |

각 단계는 난도를 다섯 수준으로 올립니다. 기본적으로 **검증 성공률 80% 이상을 두 번 연속** 달성하고 해당 수준에서 10만 스텝 이상 연습하면 다음 수준으로 넘어갑니다. 물체는 모든 단계에서 5kg입니다. 초기 단계에서는 열린 집게를 물체 가까이에 배치하며, 움직임 중에는 자동 잡기·IK·물체 부착을 사용하지 않습니다.

1단계 성공은 실제 접촉으로 잡아 테이블에서 물체 바닥을 3cm 이상, 0.2초간 들어 올린 뒤 0.5초간 잡고 유지하는 것입니다. 2~4단계는 들어 옮긴 뒤 집게를 열고 목표 영역에 1초간 안정적으로 놓아야 성공합니다. 목표 밖 테이블에 다시 놓으면 운반 자격을 다시 얻어야 합니다.

총 1억 스텝은 8개 환경을 합한 전체 예산입니다. 기준에 미달하면 그 수준을 계속 연습하므로 예산 내에 모든 단계를 통과한다는 보장은 없습니다. 4단계까지 통과하면 남은 예산으로 전체 랜덤 환경의 힘 사용을 개선합니다. 마지막 평가는 현재 단계와 관계없이 **4단계 전체 과제**에서 수행합니다.

`--viewer`는 로컬 창에 실제 학습 환경 중 첫 번째를 표시합니다. 웹 갱신은 `--live-url`·`--live-stream-port`와 비공개 인증 정보, 실행 중인 터널이 추가로 필요합니다. 창을 닫아도 학습은 계속됩니다. 검증 중에는 화면이 멈추고 터미널에 평가 진행률이 나옵니다.

## 한 단계만 학습하기

`--stage 1`을 지정하면 1단계 안에서 난도를 높이고, 2단계로는 넘어가지 않습니다.

```bash
./run_curriculum.sh --stage 1 --total-steps 100000000 --device cuda --run-dir runs/stage1_100m
```

현재 실행은 **1단계 누적 2천만 스텝 → 2단계 추가 2천만 스텝**으로 변경했습니다. 같은 모델·옵티마이저·정규화 상태를 유지하며 누적 4천만 스텝에서 종료합니다. 2단계를 통과해도 3단계로는 자동 전환하지 않습니다. 기존 실행 폴더 이름은 `runs/stage1_100m`을 그대로 사용합니다.

아래는 현재 설정으로 이어가는 명령입니다. 2단계로 이미 넘어간 뒤에는 `--stage`, `--total-steps`, `--next-stage`, `--next-stage-steps`를 생략하면 저장된 일정대로 재개합니다.

```bash
./run_curriculum.sh --resume runs/stage1_100m/latest.pt --stage 1 --total-steps 20000000 --next-stage 2 --next-stage-steps 20000000 --record-every 5000000 --record-fps 50 --record-episodes 5 --device cuda
```

`--next-stage-steps`는 다음 단계의 **추가** 예산입니다. 단계 전환 직전 `stage1_final.pt`, 전환 직후 `stage2_initial.pt`와 `stage_handoffs.jsonl`을 저장합니다. 전환은 성공률 기준과 별도로 정해진 스텝에 실행됩니다.

로그는 `runs/stage1_100m/console.log`, PID는 `runs/stage1_100m/train.pid`, 모델은 같은 폴더의 `latest.pt`에 저장합니다.

[실시간 학습 화면](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site)에서 실제 학습 환경 0번의 화면을 연속 실시간 영상으로 볼 수 있습니다. 진행 지표는 약 1초 간격으로 갱신됩니다. 본인 ChatGPT 계정으로 로그인하면 다른 네트워크에서도 볼 수 있습니다. 학습 PC가 켜져 있고 네트워크에 연결되어 있어야 갱신됩니다. 최근 업로드 상태는 `live_health.json`에 남깁니다.

실시간 전송은 `--live-url`과 별도의 비공개 서비스 자격 증명을 사용합니다. 자격 증명은 학습 프로세스에 메모리로 전달하며 코드·설정 JSON·로그에 저장하지 않습니다. 전송 장애는 학습을 막지 않고 재시도합니다.

## 500만 스텝마다 로컬 영상 저장

현재 실행에는 `--record-every 5000000 --record-fps 50 --record-episodes 5`를 적용했습니다. 해당 스텝의 모델을 고정해 평가 시나리오 5개를 **50fps, 시뮬레이션 시간 기준 1배속**으로 저장합니다. 실제 학습 환경의 연속 화면은 웹에서, 저장된 모델의 재생 영상은 로컬에서 확인합니다.

파일은 `runs/stage1_100m/recordings/step_020000000/` 같은 폴더에 저장합니다.

- `checkpoint.pt`: 해당 시점의 모델과 학습 상태
- `video.mp4`: 50fps 평가 재생 영상
- `metadata.json`: 실제 저장 스텝, 프레임 수, 성공 여부, 시드 등

처음 녹화를 설정할 때 14,776,320스텝 모델을 `initial_014776320/`에 저장했습니다. 500만 간격으로 변경할 때는 이미 1,500만 스텝을 지났으므로 다음 정기 저장은 **2천만, 2,500만, 3천만, 3,500만, 4천만** 스텝입니다. 지난 시점의 영상을 소급 생성하지 않습니다. 2천만 영상은 1단계 종료 모델이며 이후 영상은 2단계 모델입니다. 인코딩은 별도 프로세스에서 진행합니다.

웹 연속 영상은 인증된 로컬 영상 소스를 [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)로 연결하고, 기존 본인 계정 전용 웹페이지가 중계합니다. 영상 연결 프로세스 PID는 `stream_tunnel.pid`에 기록합니다. 로컬 영상 소스와 웹 중계의 인증 정보는 코드와 로그에 저장하지 않습니다.

## 중단·재개·평가

`Ctrl+C`를 누르면 현재 업데이트를 마치고 저장한 뒤 종료합니다. 이어서 학습하려면:

```bash
./run_curriculum.sh --resume runs/curriculum_100m/latest.pt --total-steps 100000000 --device cuda --viewer
```

현재 단계·난도·비용 비율과 환경·난수 상태까지 복원됩니다. `--total-steps`는 누적 목표입니다. 이전 `runs/ppo_100m` 모델은 관측·행동 방식이 달라 새 커리큘럼에 이어 붙이지 않습니다.

현재 체크포인트의 단계에서 평가하려면:

```bash
./run_evaluation.sh --checkpoint runs/curriculum_100m/best.pt --episodes 50 --device cuda --viewer
```

4단계 최종 과제로 평가하려면:

```bash
./run_evaluation.sh --checkpoint runs/curriculum_100m/best.pt --stage 4 --episodes 200 --device cuda --viewer
```

- `latest.pt`: 최근 정책·옵티마이저·정규화·커리큘럼·난수·환경 상태
- `best.pt`: 현재 난도에서 검증한 최고 정책. 난도 전환 직후에는 이전 난도에서 넘겨받은 정책이 초기 후보로 저장됩니다.
- `stage1_best.pt` ~ `stage4_best.pt`: 각 단계에서 도달한 가장 높은 난도의 최고 검증 정책
- `metrics.csv`: 단계·난도·성공률·잡기율·들기율·힘 비용
- `curriculum_history.jsonl`: 난도·단계 전환 이력
- `validation_*.json`, `final_evaluation.json`: 검증 및 최종 평가 기록

검증은 기본 10만 스텝마다 50개, 마지막 평가는 별도 시드 200개입니다. 성공을 익힌 후 4단계에서는 실제 관절 토크와 집게 힘의 절댓값을 시간 적분해 비용을 줍니다. 정지 상태에서 버티는 힘도 포함합니다. 두 번째 팔 참여 비용은 최대 −10, 운반 성공 보너스는 +100, 낙하 비용은 −25입니다.

## 수동 시뮬레이터

```bash
./run_simulator.sh --seed 31 --mass 5
```

`N`: 새 배치, `R`: 초기화, `Space`: 일시정지. Manual joints 탭에서 관절과 집게를 조작할 수 있습니다. 기존 절대 관절 목표 방식의 단일 과제 학습은 `run_training.sh`로도 실행할 수 있습니다.

## 웹에서 기록 보기

[현재 정책의 보상 상위 10개](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/replays.html)에서 영상 생성 시점에 저장한 모델의 평가 50개 중 상위 10개를 볼 수 있습니다. 모델 스텝·단계·성공 여부를 함께 표시하며, 본인 ChatGPT 계정으로 로그인하면 다른 네트워크에서도 볼 수 있습니다. 1단계는 좌우 기본 배치만 사용하므로 같은 동작이 반복될 수 있습니다. 영상은 자동 갱신되지 않으며 실시간 학습 화면과 구분합니다. [이전 단일 과제 정책 영상](https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/previous-replays.html)도 보존합니다.

저장한 정책을 새로 평가해 상위 10개 영상 파일을 만들려면 다음을 실행합니다. FFmpeg가 필요하고, 웹 반영에는 다시 게시가 필요합니다.

```bash
python -B export_replay.py --checkpoint runs/curriculum_100m/stage4_best.pt --device cpu --evaluate --eval-episodes 50 --top-k 10 --fps 50 --output runs/replay_export
```

자동 검사는 `python -m unittest discover -s tests -v`로 실행합니다.
