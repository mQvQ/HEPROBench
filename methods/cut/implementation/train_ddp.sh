#!/bin/bash

# CUT DDP Training Script
# Usage: bash train_ddp.sh <train_options> <num_gpus> [master_port] [devices]
#   train_options: all training arguments (e.g., "--dataroot /path/to/data --name experiment_name")
#   devices: comma-separated GPU IDs (e.g., "0,1" or "0,2,4,6"). If not provided, uses 0,1,2,...,num_gpus-1

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 \"<train_options>\" <num_gpus> [master_port] [devices]"
    echo "Example: $0 \"--dataroot /path/to/data --name exp1 --batch_size 4\" 2"
    echo "Example: $0 \"--dataroot /path/to/data --name exp1 --batch_size 4\" 2 29500"
    echo "Example: $0 \"--dataroot /path/to/data --name exp1 --batch_size 4\" 2 29500 \"0,2\""
    exit 1
fi

TRAIN_OPTIONS=$1
NUM_GPUS=$2
MASTER_PORT=$3
DEVICES=$4

echo "Starting DDP training with $NUM_GPUS GPUs"
echo "Train options: $TRAIN_OPTIONS"

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
# CRITICAL: Use torchrun's --master_port parameter to directly specify the port
# This is the most reliable way to ensure the port is used
if [ -n "$MASTER_PORT" ]; then
    export MASTER_PORT=$MASTER_PORT
    echo "Setting MASTER_PORT: $MASTER_PORT"
    TORCHRUN_MASTER_PORT_ARG="--master_port=$MASTER_PORT"
else
    TORCHRUN_MASTER_PORT_ARG=""
fi

# Launch DDP training using torchrun (recommended for PyTorch >= 1.10)
# Use --master_port parameter to directly specify the port (most reliable method)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    $TORCHRUN_MASTER_PORT_ARG \
    train_ddp.py \
    $TRAIN_OPTIONS

echo "DDP training completed!"

