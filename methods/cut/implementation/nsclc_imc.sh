CUDA_VISIBLE_DEVICES=5 python train.py \
--dataroot /path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches \
--name nsclc-imc-cut-repeat-b1 \
--CUT_mode cut \
--direction AtoB \
--dataset_mode NSCLCIMCREPEAT \
--preprocess crop \
--crop_size 256 \
--input_nc 26 \
--output_nc 26 \
--batch_size 1

# inference
python test_nsclc.py \
--dataroot /path/to/preprocess/data/nsclc-imc/nsclc-imc-reg-patches \
--name nsclc-imc-cut-repeat-b1 \
--CUT_mode cut \
--direction AtoB \
--dataset_mode NSCLCIMCREPEAT \
--crop_size 256 \
--input_nc 26 \
--output_nc 26 \
--checkpoints_dir /path/to/CUT/checkpoints/ \
--eval