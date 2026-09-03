# mihc-panel-1
CUDA_VISIBLE_DEVICES=3 python train.py \
--dataroot /path/to/data/r20-smu-mihc \
--name smu-mihc-cut-repeat-padding-panel-1-n-epoch-15-decay-0 \
--panel_key panel-1 \
--CUT_mode cut \
--direction AtoB \
--dataset_mode SMUMIHCREPEAT \
--preprocess corp \
--crop_size 256 \
--input_nc 7 \
--output_nc 7 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip 

CUDA_VISIBLE_DEVICES=2 python train.py \
--dataroot /path/to/data/r20-smu-mihc \
--name smu-mihc-cut-repeat-padding-panel-2-n-epoch-15-decay-0 \
--panel_key panel-2 \
--CUT_mode cut \
--direction AtoB \
--dataset_mode SMUMIHCREPEAT \
--preprocess corp \
--crop_size 256 \
--input_nc 7 \
--output_nc 7 \
--batch_size 8 \
--n_epochs 15 \
--n_epochs_decay 0 \
--no_flip

python test_orion.py \
--dataroot /path/to/data/mIHC \
--panel_key panel-2 \
--name smu-mihc-cut-repeat-padding-panel-2-n-epoch-15-decay-0 \
--CUT_mode cut --direction AtoB --dataset_mode SMUMIHCREPEAT \
--input_nc 7 --output_nc 7 --crop_size 256 \
--checkpoints_dir /path/to/CUT/checkpoints --eval --epoch 15

CUDA_VISIBLE_DEVICES=7 python test_orion.py \
--dataroot /path/to/data/mIHC \
--panel_key panel-1 \
--name smu-mihc-cut-repeat-padding-panel-1-n-epoch-15-decay-0 \
--CUT_mode cut --direction AtoB --dataset_mode SMUMIHCREPEAT \
--input_nc 7 --output_nc 7 --crop_size 256 \
--checkpoints_dir /path/to/CUT/checkpoints --eval --epoch 15