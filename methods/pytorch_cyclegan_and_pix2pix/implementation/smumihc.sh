# smu-mihc-panel-1
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /path/to/data/r20-smu-mihc \
--name smu-mihc-pix2pix-panel-1 \
--model pix2pix \
--direction AtoB \
--dataset_mode SMUMIHC \
--preprocess corp \
--crop_size 256 \
--output_nc 7 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip \
--use_wandb \
--panel_key panel-1 \
--wandb_project_name CycleGAN-and-pix2pix-smu-mihc 

# smu-mihc-panel-2
CUDA_VISIBLE_DEVICES=4 python train.py \
--dataroot /path/to/data/r20-smu-mihc \
--name smu-mihc-pix2pix-panel-2 \
--model pix2pix \
--direction AtoB \
--dataset_mode SMUMIHC \
--preprocess corp \
--crop_size 256 \
--output_nc 7 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip \
--use_wandb \
--panel_key panel-2 \
--wandb_project_name CycleGAN-and-pix2pix-smu-mihc 

# cycle-gan panel-1
CUDA_VISIBLE_DEVICES=5 python train.py \
--dataroot /path/to/data/r20-smu-mihc \
--name smu-mihc-cycle-gan-panel-1-lambda-0.0 \
--model cycle_gan \
--direction AtoB \
--dataset_mode SMUMIHC \
--preprocess corp \
--crop_size 256 \
--output_nc 7 \
--lambda_identity 0.0 \
--n_epochs 15 \
--n_epochs_decay 0 \
--batch_size 8 \
--use_wandb \
--panel_key panel-1 \
--wandb_project_name CycleGAN-and-pix2pix-smu-mihc 

# smu-mihc-cycle-gan panel-2 lambda-0.1
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot /path/to/data/r20-smu-mihc \
--name smu-mihc-cycle-gan-panel-2-lambda-0.1 \
--model cycle_gan \
--direction AtoB \
--dataset_mode SMUMIHC \
--preprocess corp \
--crop_size 256 \
--output_nc 7 \
--lambda_identity 0.0 \
--n_epochs 15 \
--n_epochs_decay 0 \
--batch_size 8 \
--use_wandb \
--panel_key panel-2 \
--wandb_project_name CycleGAN-and-pix2pix-smu-mihc 