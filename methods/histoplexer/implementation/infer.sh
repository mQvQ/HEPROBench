python -m bin.inference --checkpoint_path="${HISTOPLEXER_CHECKPOINT_PATH:?Set HISTOPLEXER_CHECKPOINT_PATH to an authorized checkpoint}" \
                        --get_predictions \
                        --src_folder="${HISTOPLEXER_HE_ROOT:?Set HISTOPLEXER_HE_ROOT to an authorized H&E directory}" \
                        --tgt_folder="${HISTOPLEXER_TARGET_ROOT:?Set HISTOPLEXER_TARGET_ROOT to an authorized multiplex directory}"
# train
python -m bin.train --config_path="${HISTOPLEXER_CONFIG_PATH:?Set HISTOPLEXER_CONFIG_PATH to a local training config}"
