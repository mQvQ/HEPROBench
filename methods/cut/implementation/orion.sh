python test_orion.py --dataroot /path/to/MIPHEI-ViT \
--name crc-orion-cut-zero-padding \
--CUT_mode cut \
--direction AtoB \
--dataset_mode CRCORIONPADDING \
--crop_size 256 \
--input_nc 16 \
--output_nc 16 \
--checkpoints_dir /path/to/CUT/checkpoints/ \
--eval  

python test_orion.py --dataroot /path/to/MIPHEI-ViT \
--name crc-orion-cut-repeat-padding \
--CUT_mode cut \
--direction AtoB \
--dataset_mode CRCORIONREPEAT \
--crop_size 256 \
--input_nc 16 \
--output_nc 16 \
--checkpoints_dir /path/to/CUT/checkpoints/ \
--eval 
