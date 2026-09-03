# orion-pix2pix
CUDA_VISIBLE_DEVICES=7 python train.py \
--dataroot /path/to/MIPHEI-ViT \
--name crc-orion-pix2pix \
--model pix2pix \
--direction AtoB \
--dataset_mode CRCORION \
--preprocess corp \
--crop_size 256 \
--output_nc 16 \
--batch_size 8 \
--n_epochs 10 \
--no_flip \
--use_wandb \
--wandb_project_name CycleGAN-and-pix2pix-orion \
--save_by_iter 
# orion: cycle-gan
CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /path/to/MIPHEI-ViT \
--name crc-orion-cycle_gan \
--model cycle_gan \
--direction AtoB \
--dataset_mode CRCORION \
--preprocess corp \
--crop_size 256 \
--output_nc 16 \
--batch_size 8 \
--n_epochs 10 \
--no_flip \
--use_wandb \
--wandb_project_name CycleGAN-and-pix2pix-orion \
--save_by_iter \
--lambda_identity 0.0

# test orion pix2pix
python test_orion.py --dataroot /path/to/MIPHEI-ViT --name crc-orion-pix2pix --model pix2pix --direction AtoB --dataset_mode CRCORION  \
--output_nc 16 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --load_iter 100000
# test orion cycle_gan
python test_orion.py --dataroot /path/to/MIPHEI-ViT --name crc-orion-cycle_gan --model cycle_gan --direction AtoB --dataset_mode CRCORION  \
--output_nc 16 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --load_iter 100000
# test orion pix2pix
python test_orion.py --dataroot /path/to/MIPHEI-ViT --name crc-orion-pix2pix --model pix2pix --direction AtoB --dataset_mode CRCORION  \
--output_nc 16 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --load_iter 2000000

# 0803; conda env: image-gen

CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot /path/to/MIPHEI-ViT \
--name crc-orion-cycle_gan-n-epoch-15-decay-0-lambda-0.0 \
--model cycle_gan \
--direction AtoB \
--dataset_mode CRCORION \
--preprocess corp \
--crop_size 256 \
--output_nc 16 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip \
--use_wandb \
--wandb_project_name CycleGAN-and-pix2pix-orion \
--lambda_identity 0.0

# nsclc-imc
CUDA_VISIBLE_DEVICES=4 python train.py \
--dataroot /path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches \
--name nsclc-imc-iter-1000000-b-1 \
--model cycle_gan \
--direction AtoB \
--dataset_mode NSCLCIMCREPEAT \
--preprocess corp \
--crop_size 256 \
--input_nc 26 \
--output_nc 26 \
--batch_size 1 \
--no_flip \
--lambda_identity 0.0
