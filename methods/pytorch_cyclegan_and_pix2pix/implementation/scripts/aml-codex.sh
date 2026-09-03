# aml-codex
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /path/to/data/AML-CODEX-preprocess \
--name AML-CODEX-pix2pix \
--model pix2pix \
--direction AtoB \
--dataset_mode AMLCODEX \
--preprocess corp \
--crop_size 256 \
--output_nc 54 \
--batch_size 16 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip
# cycle-gan
CUDA_VISIBLE_DEVICES=3 python train.py \
--dataroot /path/to/data/AML-CODEX-preprocess \
--name AML-CODEX-cyclegan \
--model cycle_gan \
--direction AtoB \
--dataset_mode AMLCODEX \
--preprocess corp \
--crop_size 256 \
--output_nc 54 \
--lambda_identity 0.0 \
--n_epochs 15 \
--n_epochs_decay 0 \
--batch_size 8 \
--use_wandb \
--wandb_project_name CycleGAN-and-pix2pix-aml-codex
