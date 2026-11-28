#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/conda_python.sh
task_run_dir="${REACHER_STUDY_RUN_DIR:-runs/studies/three_models_1kg_v3}"
exec systemd-inhibit --what=sleep --mode=block --why="PPO A2C SAC training" \
  "$python_bin" -B -m training.compare \
  --run-dir "$task_run_dir" --algorithms ppo a2c sac --parallel 3 \
  --web --connection-dir runtime/web_connection \
  --stage 1 --level 0 --layout-mode progressive \
  --auto-stages --min-level-steps 0 \
  --stage-max-steps 10000000 --total-steps 40000000 --device cpu \
  --record-every 5000000 --record-episodes 5 "$@"
