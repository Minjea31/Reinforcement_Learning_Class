#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/conda_python.sh

default_study_dir=runs/studies/three_models_1kg_masked
mode=(--no-auto-levels --total-steps 65536 --eval-every 16384 --record-every 0)
if [[ "${1:-}" == --full ]]; then
  shift
  default_study_dir=runs/studies/three_models_1kg_masked_full
  mode=(--auto-stages --min-level-steps 0 --stage-max-steps 10000000 --total-steps 40000000 --eval-every 24576 --record-every 5000000)
fi
study_dir="${REACHER_MASKED_STUDY_DIR:-$default_study_dir}"
export CUDA_VISIBLE_DEVICES=
exec systemd-inhibit --what=sleep --mode=block --why="Masked reinforcement learning comparison" \
  nice -n 15 ionice -c 3 "$python_bin" -B -m training.compare \
  --run-dir "$study_dir" --algorithms ppo a2c sac --parallel 3 \
  --web --local-viewer --connection-dir runtime/web_connection \
  --stage 1 --level 0 --layout-mode progressive --payload-mass 1 \
  --fixed-study-normalization "$study_dir/fixed_normalization.json" \
  --rollout-steps 2048 --device cpu --eval-episodes 50 "${mode[@]}" "$@"
