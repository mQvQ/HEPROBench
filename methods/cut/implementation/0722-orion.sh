# orion-cut-repeat padding
CUDA_VISIBLE_DEVICES=3 python train.py \
--dataroot /path/to/MIPHEI-ViT \
--name crc-orion-cut-repeat-padding \
--CUT_mode cut \
--direction AtoB \
--dataset_mode CRCORIONREPEAT \
--preprocess corp \
--crop_size 256 \
--input_nc 16 \
--output_nc 16 \
--batch_size 8 \
--n_epochs 15 \
--no_flip 
 
# orion-cut zero padding
CUDA_VISIBLE_DEVICES=1 python train.py \
--dataroot /path/to/MIPHEI-ViT \
--name crc-orion-cut-zero-padding \
--CUT_mode cut \
--direction AtoB \
--dataset_mode CRCORIONPADDING \
--preprocess corp \
--crop_size 256 \
--input_nc 16 \
--output_nc 16 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip 

# orion-cut conditionlayer