python test_orion.py --dataroot /path/to/data/mIHC --name smu-mihc-pix2pix-panel-2 \
 --model pix2pix --direction AtoB \
--dataset_mode SMUMIHC --panel_key panel-2 --phase infer_valid_test \
--output_nc 7 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15


python test_orion.py --dataroot /path/to/data/mIHC --name smu-mihc-pix2pix-panel-1 \
 --model pix2pix --direction AtoB \
--dataset_mode SMUMIHC --panel_key panel-1 --phase infer_valid_test \
--output_nc 7 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15



python test_orion.py --dataroot /path/to/data/mIHC --name smu-mihc-cycle-gan-panel-1-lambda-0.0 \
 --model cycle_gan --direction AtoB \
--dataset_mode SMUMIHC --panel_key panel-1 --phase infer_valid_test \
--output_nc 7 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15

python test_orion.py --dataroot /path/to/data/mIHC --name smu-mihc-cycle-gan-panel-2-lambda-0.1 \
 --model cycle_gan --direction AtoB \
--dataset_mode SMUMIHC --panel_key panel-2 --phase infer_valid_test \
--output_nc 7 --crop_size 256 \
--checkpoints_dir /path/to/pytorch-CycleGAN-and-pix2pix/checkpoints/ --eval --epoch 15