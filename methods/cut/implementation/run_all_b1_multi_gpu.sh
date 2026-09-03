#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/path/to/benchmark/results/CUT"
LOG_DIR="$RESULTS_DIR/logs_b1"
TOTAL_ITERATIONS=${1:-100000}
NUM_THREADS_VALUE="${NUM_THREADS:-4}"

mkdir -p "$LOG_DIR"

launch_experiment() {
    local exp_key=$1
    local gpu_id=$2
    local session_name=$3
    local result_name=$4
    local log_path="$LOG_DIR/${session_name}.log"
    local resume_iter=0

    if [ -d "$RESULTS_DIR/$result_name" ]; then
        resume_iter=$(find "$RESULTS_DIR/$result_name" -maxdepth 1 -type f -name 'iter_*_net_G.pth' \
            | sed -E 's#^.*/iter_([0-9]+)_net_G\.pth$#\1#' \
            | sort -n \
            | tail -n 1)
        resume_iter=${resume_iter:-0}
    fi

    local extra_args=()
    if [ "$resume_iter" -gt 0 ]; then
        extra_args+=(--continue_train --continue_iteration "$resume_iter")
    fi

    echo "Launching $exp_key on GPU $gpu_id (resume_iter=$resume_iter) -> $log_path"
    screen -dmS "$session_name" /bin/bash -lc \
        "cd '$SCRIPT_DIR' && NUM_THREADS='$NUM_THREADS_VALUE' bash '$SCRIPT_DIR/run_b1_experiment.sh' '$exp_key' '$gpu_id' '$TOTAL_ITERATIONS' ${extra_args[*]} > '$log_path' 2>&1"
}

launch_experiment hemit 0 cut_b1_hemit hemit-cut-repeat-b1
launch_experiment aml 1 cut_b1_aml aml-codex-cut-repeat-b1
launch_experiment crc 2 cut_b1_crc crc-codex-cut-repeat-b1
launch_experiment nsclc 3 cut_b1_nsclc nsclc-imc-cut-repeat-b1
launch_experiment smu-p1 4 cut_b1_smu_p1 smu-p1-cut-repeat-b1
launch_experiment smu-p2 5 cut_b1_smu_p2 smu-p2-cut-repeat-b1
launch_experiment mt 7 cut_b1_mt multi-tumor-codex-cut-repeat-b1

echo "Launched all CUT b1 experiments."
