#!/bin/bash

# HistoPlexer DDP Training Script
# Usage: bash train_ddp.sh <config_path> <num_gpus> [master_port] [devices]
#   devices: comma-separated GPU IDs (e.g., "0,1" or "0,2,4,6"). If not provided, uses 0,1,2,...,num_gpus-1

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_path> <num_gpus> [master_port] [devices]"
    echo "Example: $0 src/config/sample_config_ddp.json 4"
    echo "Example: $0 src/config/sample_config_ddp.json 4 29500"
    echo "Example: $0 src/config/sample_config_ddp.json 2 29500 \"0,2\""
    exit 1
fi

CONFIG_PATH=$1
NUM_GPUS=$2
MASTER_PORT=$3
DEVICES=$4

echo "Starting DDP training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Change to the script directory
cd "$SCRIPT_DIR"

# Set CUDA_VISIBLE_DEVICES
if [ -n "$DEVICES" ]; then
    # Use specified devices
    export CUDA_VISIBLE_DEVICES=$DEVICES
    echo "Using specified GPUs: $CUDA_VISIBLE_DEVICES"
else
    # Use default sequential devices starting from 0
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Using default GPUs: $CUDA_VISIBLE_DEVICES"
fi

# Set master port if provided
if [ -n "$MASTER_PORT" ]; then
    export MASTER_PORT=$MASTER_PORT
    MASTER_PORT_ARG="--master_port=$MASTER_PORT"
    echo "Using master port: $MASTER_PORT"
else
    MASTER_PORT_ARG=""
    echo "Using default master port: 29500"
fi

# Launch DDP training using torchrun (recommended for PyTorch >= 1.10)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    $MASTER_PORT_ARG \
    -m bin.train_ddp \
    --config_path="$CONFIG_PATH"

echo "DDP training completed!"
