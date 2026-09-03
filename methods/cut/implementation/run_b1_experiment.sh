#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_key> [cut|fastcut] [gpu_id] [total_iterations] [extra_train_args...]"
    echo "Example: $0 smu-p2 fastcut 0 100000 --print_freq 1 --save_latest_freq 10"
    exit 1
fi

EXPERIMENT_KEY=$1
METHOD_VARIANT="cut"
if [ $# -ge 2 ]; then
    case "${2,,}" in
        cut|fastcut)
            METHOD_VARIANT="${2,,}"
            shift
            ;;
    esac
fi

GPU_ID=${2:-0}
TOTAL_ITERATIONS=${3:-100000}
TRAIN_GPU_ID=0

if [ $# -ge 3 ]; then
    shift 3
else
    shift 1
    if [ $# -gt 0 ]; then
        shift 1 || true
    fi
    if [ $# -gt 0 ]; then
        shift 1 || true
    fi
fi

EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/miniconda3/envs/image-gen/bin/python}"
CHECKPOINTS_DIR="/path/to/benchmark/results/CUT"
NUM_THREADS="${NUM_THREADS:-4}"
METHOD_NAME_UPPER="CUT"
N_EPOCHS=200
N_EPOCHS_DECAY=200
N_ITERS=0
N_ITERS_DECAY=0

NAME=""
DATAROOT=""
DATASET_MODE=""
INPUT_NC=""
OUTPUT_NC=""
PREPROCESS=""
CUT_MODE=""
PANEL_KEY="panel-1"
SAVE_BY_ITER=0

case "$METHOD_VARIANT" in
    cut)
        METHOD_NAME_UPPER="CUT"
        N_EPOCHS=200
        N_EPOCHS_DECAY=200
        N_ITERS=$(((TOTAL_ITERATIONS + 1) / 2))
        N_ITERS_DECAY=$((TOTAL_ITERATIONS - N_ITERS))
        ;;
    fastcut)
        METHOD_NAME_UPPER="FastCUT"
        N_EPOCHS=150
        N_EPOCHS_DECAY=50
        N_ITERS=$(((TOTAL_ITERATIONS * 3 + 3) / 4))
        N_ITERS_DECAY=$((TOTAL_ITERATIONS - N_ITERS))
        ;;
    *)
        echo "Unknown method variant: $METHOD_VARIANT"
        echo "Supported methods: cut fastcut"
        exit 1
        ;;
esac

case "$EXPERIMENT_KEY" in
    aml)
        NAME="aml-codex-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/preprocess/data/aml-codex-preprocess"
        DATASET_MODE="HE2SPREPEAT"
        INPUT_NC=54
        OUTPUT_NC=54
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        ;;
    crc)
        NAME="crc-codex-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/preprocess/data/crc-codex-new/crc-codex-reg-patches-new-v2"
        DATASET_MODE="HE2SPREPEAT"
        INPUT_NC=58
        OUTPUT_NC=58
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        ;;
    hemit)
        NAME="hemit-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/preprocess/data/HEMIT_20x_patches"
        DATASET_MODE="HE2SPREPEAT"
        INPUT_NC=3
        OUTPUT_NC=3
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        SAVE_BY_ITER=1
        ;;
    mt)
        NAME="multi-tumor-codex-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches"
        DATASET_MODE="HE2SPREPEAT"
        INPUT_NC=60
        OUTPUT_NC=60
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        ;;
    nsclc)
        NAME="nsclc-imc-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches"
        DATASET_MODE="HE2SPREPEAT"
        INPUT_NC=28
        OUTPUT_NC=28
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        ;;
    smu-p1)
        NAME="smu-p1-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/data/mIHC"
        DATASET_MODE="SMUMIHCREPEAT"
        INPUT_NC=7
        OUTPUT_NC=7
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        PANEL_KEY="panel-1"
        ;;
    smu-p2)
        NAME="smu-p2-${METHOD_VARIANT}-repeat-b1"
        DATAROOT="/path/to/data/mIHC"
        DATASET_MODE="SMUMIHCREPEAT"
        INPUT_NC=7
        OUTPUT_NC=7
        PREPROCESS="crop"
        CUT_MODE="$METHOD_NAME_UPPER"
        PANEL_KEY="panel-2"
        ;;
    *)
        echo "Unknown experiment_key: $EXPERIMENT_KEY"
        echo "Supported keys: aml crc hemit mt nsclc smu-p1 smu-p2"
        exit 1
        ;;
esac

COMMON_ARGS=(
    --dataroot "$DATAROOT"
    --name "$NAME"
    --CUT_mode "$CUT_MODE"
    --direction AtoB
    --checkpoints_dir "$CHECKPOINTS_DIR"
    --dataset_mode "$DATASET_MODE"
    --preprocess "$PREPROCESS"
    --crop_size 256
    --input_nc "$INPUT_NC"
    --output_nc "$OUTPUT_NC"
    --batch_size 1
    --total_iterations "$TOTAL_ITERATIONS"
    --n_epochs "$N_EPOCHS"
    --n_epochs_decay "$N_EPOCHS_DECAY"
    --n_iters "$N_ITERS"
    --n_iters_decay "$N_ITERS_DECAY"
    --display_freq 400
    --display_id -1
    --print_freq 100
    --save_latest_freq 5000
    --save_per_iteration 10000
    --num_threads "$NUM_THREADS"
    --panel_key "$PANEL_KEY"
    --gpu_ids "$TRAIN_GPU_ID"
)

case "$METHOD_VARIANT" in
    cut)
        COMMON_ARGS+=(
            --nce_idt true
            --lambda_GAN 1.0
            --lambda_NCE 1.0
            --flip_equivariance false
        )
        ;;
    fastcut)
        COMMON_ARGS+=(
            --nce_idt false
            --lambda_GAN 1.0
            --lambda_NCE 10.0
            --flip_equivariance true
        )
        ;;
esac

if [ "$SAVE_BY_ITER" -eq 1 ]; then
    COMMON_ARGS+=(--save_by_iter)
fi

cd "$SCRIPT_DIR"

echo "Running $NAME on GPU $GPU_ID"
echo "Python: $PYTHON_BIN"
echo "Dataroot: $DATAROOT"
echo "Total iterations: $TOTAL_ITERATIONS"
echo "Method variant: $METHOD_VARIANT"
echo "CUT_mode: $CUT_MODE"
echo "n_epochs/n_epochs_decay: $N_EPOCHS/$N_EPOCHS_DECAY"
echo "n_iters/n_iters_decay: $N_ITERS/$N_ITERS_DECAY"
echo "Num threads: $NUM_THREADS"
echo "Command:"
printf '  %q' "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
printf '\n'

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
