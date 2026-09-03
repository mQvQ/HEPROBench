#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /path/to/preprocess/data/crc-codex-new/crc-codex-reg-patches-new \
--name CRC-CODEX-pix2pix \
--model pix2pix \
--direction AtoB \
--dataset_mode CRCCODEX \
--preprocess corp \
--crop_size 256 \
--output_nc 58 \
--batch_size 16 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip
&
CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /path/to/preprocess/data/crc-codex-new/crc-codex-reg-patches-new \
--name CRC-CODEX-cyclegan \
--model cycle_gan \
--direction AtoB \
--dataset_mode CRCCODEX \
--preprocess corp \
--crop_size 256 \
--output_nc 58 \
--lambda_identity 0.0 \
--n_epochs 15 \
--n_epochs_decay 0 \
--batch_size 16 \
