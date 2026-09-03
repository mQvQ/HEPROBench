#!/usr/bin/env bash

#python test_orion.py --dataroot /path/to/data/AML-CODEX-preprocess --name AML-CODEX-pix2pix \
# --model pix2pix --direction AtoB \
#--dataset_mode AMLCODEX --phase infer_valid_test \
#--output_nc 54 --crop_size 256 \
#--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15
#
#python test_orion.py --dataroot /path/to/data/AML-CODEX-preprocess --name AML-CODEX \
# --model cycle_gan --direction AtoB \
#--dataset_mode AMLCODEX --phase infer_valid_test \
#--output_nc 54 --crop_size 256 \
#--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15

python test_orion.py --dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches --name MULTITUMOR-CODEX-pix2pix \
 --model pix2pix --direction AtoB \
--dataset_mode MULTITUMORCODEX --phase infer_valid_test \
--output_nc 51 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15

python test_orion.py --dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches --name MULTITUMOR-CODEX-cyclegan \
 --model cycle_gan --direction AtoB \
--dataset_mode MULTITUMORCODEX --phase infer_valid_test \
--output_nc 51 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15