# infer crc-codex-tma
CUDA_VISIBLE_DEVICES=6 python test_orion.py \
--dataroot /path/to/preprocess/data/crc-codex-new/crc-codex-reg-patches-new \
--name crc-codex-cut-repeat \
--CUT_mode cut \
--direction AtoB \
--dataset_mode CRCCODEXREPEAT \
--preprocess corp \
--crop_size 256 \
--input_nc 58 \
--output_nc 58 \
--batch_size 8 \
--checkpoints_dir /path/to/CUT/checkpoints/ \
--eval

CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches --name multi-tumor-codex-cut-repeat --CUT_mode cut --direction AtoB --dataset_mode MTCODEXREPEAT --preprocess crop --crop_size 256 --input_nc 51 --output_nc 51 --batch_size 8 --n_epochs 15 --n_epochs_decay 0 --no_flip

CUDA_VISIBLE_DEVICES=6 python test_orion.py --dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches --name multi-tumor-codex-cut-repeat --CUT_mode cut --direction AtoB --dataset_mode MTCODEXREPEAT --preprocess corp --crop_size 256 --input_nc 51 --output_nc 51 --batch_size 1 --checkpoints_dir /path/to/CUT/checkpoints/ --eval
