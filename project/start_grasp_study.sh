#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/conda_python.sh

reward_scheme=grasp_progress
if [[ "${1:-}" == --lift ]]; then
    shift
    reward_scheme=lift_progress
fi
default_run_dir="runs/studies/ppo_a2c_${reward_scheme}_pilot"
budget=(--no-auto-levels --total-steps 65536 --record-every 0)
if [[ "${1:-}" == --full ]]; then
    shift
    default_run_dir="runs/studies/ppo_a2c_${reward_scheme}"
    budget=(--auto-stages --min-level-steps 0 --stage-max-steps 10000000 --total-steps 40000000
            --record-every 5000000 --record-episodes 5)
fi
study_dir="${REACHER_GRASP_RUN_DIR:-$default_run_dir}"
exec systemd-inhibit --what=sleep --mode=block --why="PPO A2C grasp training" \
  "$python_bin" -B -m training.compare \
  --run-dir "$study_dir" --algorithms ppo a2c --parallel 2 \
  --web --local-viewer --connection-dir runtime/web_connection \
  --stage 1 --level 0 --layout-mode progressive --reward-scheme "$reward_scheme" \
  --payload-mass 1 --device cpu --rollout-steps 2048 \
  "${budget[@]}" "$@"
