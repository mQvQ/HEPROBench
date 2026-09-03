CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /path/to/data/HEMIT_20x_patches \
--name hemit-pix2pix \
--model pix2pix \
--direction AtoB \
--dataset_mode HEMIT \
--preprocess corp \
--crop_size 256 \
--output_nc 3 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip


CUDA_VISIBLE_DEVICES=6 python train.py \
--dataroot /path/to/data/HEMIT_20x_patches \
--name hemit-cycle_gan \
--model cycle_gan \
--direction AtoB \
--dataset_mode HEMIT \
--preprocess corp \
--crop_size 256 \
--output_nc 3 \
--lambda_identity 0.0 \
--n_epochs 15 \
--n_epochs_decay 0 \
--batch_size 8