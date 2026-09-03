python -m bin.inference --checkpoint_path=/path/to/HistoPlexer/results/smu-p1-wo-high-res/smu_ours_channels-all_seed-96/checkpoint-step_500000.pt \
                        --get_predictions \
                        --src_folder=/data1/tma/mIHC/Series-14-After-Registration-High-Quality-Region-Cropped-Patch \
                        --tgt_folder=/data1/tma/mIHC/Series-14-After-Registration-High-Quality-Region-Cropped-Patch
# train
python -m bin.train --config_path=/path/to/HistoPlexer/src/config/smu_config_panel-2.json