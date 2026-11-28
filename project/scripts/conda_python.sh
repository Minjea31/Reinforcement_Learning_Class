#!/usr/bin/env bash
# Shared by the launchers: use the activated Conda environment.
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == base || ! -f "${CONDA_PREFIX}/conda-meta/history" ]]; then
    echo '먼저 conda activate dual-reacher 를 실행하세요.' >&2
    exit 1
fi
python_bin="${CONDA_PREFIX}/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
