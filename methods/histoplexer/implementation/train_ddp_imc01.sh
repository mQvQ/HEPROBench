#!/bin/bash

# HistoPlexer DDP Training Script (IMC scaled to 0-1)
# Usage: bash train_ddp_imc01.sh <config_path> <num_gpus> [master_port] [devices]
#   devices: comma-separated GPU IDs (e.g., "0,1" or "0,2,4,6"). If not provided, uses 0,1,2,...,num_gpus-1

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_path> <num_gpus> [master_port] [devices]"
    echo "Example: $0 src/config/mt_codex_ddp.json 4"
    echo "Example: $0 src/config/mt_codex_ddp.json 4 29500"
    echo "Example: $0 src/config/mt_codex_ddp.json 2 29500 \"0,2\""
    exit 1
fi

CONFIG_PATH=$1
NUM_GPUS=$2
MASTER_PORT=$3
DEVICES=$4

echo "Starting DDP training (IMC 0-1) with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

if [ -n "$DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
    echo "Using specified GPUs: $CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Using default GPUs: $CUDA_VISIBLE_DEVICES"
fi

if [ -n "$MASTER_PORT" ]; then
    export MASTER_PORT=$MASTER_PORT
    MASTER_PORT_ARG="--master_port=$MASTER_PORT"
    echo "Using master port: $MASTER_PORT"
else
    MASTER_PORT_ARG=""
    echo "Using default master port: 29500"
fi

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    $MASTER_PORT_ARG \
    -m bin.train_ddp_imc01 \
    --config_path="$CONFIG_PATH"

echo "DDP training (IMC 0-1) completed!"
