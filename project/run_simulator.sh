#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/conda_python.sh
exec "$python_bin" -B -m simulation.simulator "$@"
