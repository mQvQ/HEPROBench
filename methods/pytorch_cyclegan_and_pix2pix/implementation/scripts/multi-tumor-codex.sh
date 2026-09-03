#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches \
--name MULTITUMOR-CODEX-pix2pix \
--model pix2pix \
--direction AtoB \
--dataset_mode MULTITUMORCODEX \
--preprocess corp \
--crop_size 256 \
--output_nc 51 \
--batch_size 16 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip
&
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches \
--name MULTITUMOR-CODEX-cyclegan \
--model cycle_gan \
--direction AtoB \
--dataset_mode MULTITUMORCODEX \
--preprocess corp \
--crop_size 256 \
--output_nc 51 \
--lambda_identity 0.0 \
--n_epochs 15 \
--n_epochs_decay 0 \
--batch_size 16
