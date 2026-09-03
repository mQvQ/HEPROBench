#!/bin/bash

# DDP training script for nsclc-imc dataset
# Usage: bash nsclc_imc_ddp.sh <num_gpus> [master_port] [devices]

NUM_GPUS=${1:-2}  # Default to 2 GPUs
MASTER_PORT=$2
DEVICES=$3

# Build train options string
TRAIN_OPTS="--dataroot /path/to/preprocess/data/aml-codex-preprocess \
--name aml-codex-cut-repeat-b16 \
--CUT_mode cut \
--direction AtoB \
--checkpoints_dir /path/to/benchmark/results/CUT \
--dataset_mode HE2SPREPEAT \
--preprocess crop \
--crop_size 256 \
--input_nc 54 \
--output_nc 54 \
--batch_size 8"

# Always add ddp_master_port if MASTER_PORT is provided
# This ensures the port is passed even if environment variable doesn't work
if [ -n "$MASTER_PORT" ]; then
    TRAIN_OPTS="$TRAIN_OPTS --ddp_master_port $MASTER_PORT"
    echo "Setting --ddp_master_port=$MASTER_PORT in train options"
fi

bash train_ddp.sh "$TRAIN_OPTS" \
$NUM_GPUS \
$MASTER_PORT \
$DEVICES

