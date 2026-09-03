#!/bin/bash

# Simple HistoPlexer DDP Training Script using torchrun
# Usage: bash train_ddp_simple.sh <config_path> <num_gpus>

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_path> <num_gpus>"
    echo "Example: $0 src/config/sample_config_ddp.json 4"
    exit 1
fi

CONFIG_PATH=$1
NUM_GPUS=$2

echo "Starting DDP training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Change to the script directory
cd "$SCRIPT_DIR"

# Set environment variables for DDP
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Launch DDP training using torchrun (recommended for PyTorch >= 1.10)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    -m bin.train_ddp \
    --config_path="$CONFIG_PATH"

echo "DDP training completed!"
