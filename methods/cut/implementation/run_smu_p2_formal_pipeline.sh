#!/bin/bash

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cut|fastcut> <gpu_id> [total_iterations]"
    exit 1
fi

MODE="${1,,}"
GPU_ID="$2"
TOTAL_ITERATIONS="${3:-100000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/miniconda3/envs/image-gen/bin/python}"
RESULTS_DIR="/path/to/benchmark/results/CUT"
NUM_THREADS="${NUM_THREADS:-4}"
SAVE_FREQ="${SAVE_FREQ:-10000}"

case "$MODE" in
    cut)
        NAME="smu-p2-cut-align-b1-formal"
        INFER_CONFIG="/path/to/benchmark/configs/infer/cut-smu-crc-p2-align-b1.json"
        ;;
    fastcut)
        NAME="smu-p2-fastcut-align-b1-formal"
        INFER_CONFIG="/path/to/benchmark/configs/infer/fastcut-smu-crc-p2-align-b1.json"
        ;;
    *)
        echo "Unsupported mode: $MODE"
        echo "Supported modes: cut fastcut"
        exit 1
        ;;
esac

RESULT_DIR="$RESULTS_DIR/$NAME"
RESUME_ITER=0
if [ -d "$RESULT_DIR" ]; then
    RESUME_ITER=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'iter_*_net_G.pth' \
        | sed -E 's#^.*/iter_([0-9]+)_net_G\.pth$#\1#' \
        | sort -n \
        | tail -n 1)
    RESUME_ITER=${RESUME_ITER:-0}
fi

TRAIN_ARGS=(
    --name "$NAME"
    --checkpoints_dir "$RESULTS_DIR"
    --save_latest_freq "$SAVE_FREQ"
    --save_per_iteration "$SAVE_FREQ"
    --display_id -1
    --no_html
)
if [ "$RESUME_ITER" -gt 0 ]; then
    TRAIN_ARGS+=(--continue_train --continue_iteration "$RESUME_ITER")
fi

cd "$SCRIPT_DIR"

echo "[PIPELINE] mode=$MODE gpu=$GPU_ID total_iterations=$TOTAL_ITERATIONS resume_iter=$RESUME_ITER"
echo "[PIPELINE] training name=$NAME"
echo "[PIPELINE] infer config=$INFER_CONFIG"

NUM_THREADS="$NUM_THREADS" bash "$SCRIPT_DIR/run_b1_experiment.sh" smu-p2 "$MODE" "$GPU_ID" "$TOTAL_ITERATIONS" "${TRAIN_ARGS[@]}"

echo "[PIPELINE] training finished, starting inference"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" -u "$SCRIPT_DIR/sp_infer.py" --config "$INFER_CONFIG" --split both --gpu_ids 0

echo "[PIPELINE] completed mode=$MODE"
