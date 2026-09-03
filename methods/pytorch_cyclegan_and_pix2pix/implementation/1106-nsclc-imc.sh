# cyclegan
CUDA_VISIBLE_DEVICES=4 python train.py \
--dataroot /path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches \
--name nsclc-imc-iter-1000000-b1 \
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
# pix2pix

CUDA_VISIBLE_DEVICES=2 python train.py \
--dataroot /path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches \
--name nsclc-imc-pix2pix-iter-1000000-b1 \
--model pix2pix \
--direction AtoB \
--dataset_mode NSCLCIMCREPEAT \
--preprocess corp \
--crop_size 256 \
--output_nc 26 \
--input_nc 26 \
--batch_size 1 \
--no_flip


python test_npy.py --dataroot /path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches \
 --name nsclc-imc-pix2pix-iter-1000000-b1 \
 --model pix2pix --direction AtoB \
--dataset_mode NSCLCIMCREPEAT --phase infer_valid_test \
--input_nc 26 --output_nc 26 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --load_iter 1000000