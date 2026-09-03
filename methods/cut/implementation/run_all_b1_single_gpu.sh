#!/bin/bash

set -euo pipefail

GPU_ID=${1:-0}
TOTAL_ITERATIONS=${2:-100000}
NUM_THREADS_VALUE="${NUM_THREADS:-4}"

if [ $# -ge 2 ]; then
    shift 2
elif [ $# -eq 1 ]; then
    shift 1
fi

EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS=(
    hemit
    aml
    crc
    mt
    nsclc
    smu-p1
    smu-p2
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "=== Starting $exp ==="
    NUM_THREADS="$NUM_THREADS_VALUE" bash "$SCRIPT_DIR/run_b1_experiment.sh" "$exp" "$GPU_ID" "$TOTAL_ITERATIONS" "${EXTRA_ARGS[@]}"
    echo "=== Finished $exp ==="
done
