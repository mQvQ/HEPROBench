# large he region input (333) r15
python eval_orion_multi_process.py \
  --input_dir /disk2/tma/ORIONCRC_dataset_tile_20x \
  --output_dir /disk2/tma/rosie/large_region_orion \
  --model_path /path/to/rosie/best_model_single.pth \
  --split test \
  --num_gpus 4 \
  --procs_per_gpu 2
# large he region input (333) r15
CUDA_VISIBLE_DEVICES=2,3 python eval_orion_multi_process.py \
  --input_dir /disk2/tma/ORIONCRC_dataset_tile_20x \
  --output_dir /disk2/tma/rosie/large_region_orion \
  --model_path /path/to/rosie/best_model_single.pth \
  --split val \
  --num_gpus 2 \
  --procs_per_gpu 2
# center crop(输入只有256*256的视野)
CUDA_VISIBLE_DEVICES=2,3,4,5, python eval_orion_center_crop_multi_process.py \
--input_dir /data2/tma/r18-orion/ORIONCRC_dataset_tile_20x \
--output_dir /data2/tma/rosie_results/orion_256_input \
--model_path /path/to/rosie/best_model_single.pth \
--split test \
--num_gpus 4 \
--procs_per_gpu 2
# center crop (输入只有256*256的视野), valid split
CUDA_VISIBLE_DEVICES=0,1,6,7, python eval_orion_center_crop_multi_process.py \
--input_dir /data2/tma/r18-orion/ORIONCRC_dataset_tile_20x \
--output_dir /data2/tma/rosie_results/orion_256_input \
--model_path /path/to/rosie/best_model_single.pth \
--split val \
--num_gpus 4 \
--procs_per_gpu 2