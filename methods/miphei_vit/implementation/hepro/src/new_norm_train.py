"""
Training scripts for image to image segmentation.
"""
import logging
logging.getLogger('pyvips').setLevel(logging.WARNING)

import os
os.environ['VIPS_CONCURRENCY'] = '1'
import pyvips

from omegaconf import OmegaConf
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb
import torch

from .dataset import NormalizationLayer, get_augmentations, DataModule,\
                     BalancedPositiveSampler, get_width_height, get_effective_width_height,\
                     get_input_mean_std, get_new_input_mean_std
from .metrics import CellMetrics
from .models import ModelModule, DiscriminatorPatch
from .utils import wandb_log_artifact, get_foreground_weight, update_wandb_note
from .callbacks import WandbVisCallback, CustomModelCheckpoint, SlideAugentationCallback, SwitchGenDiscTrain,\
                       DebugImageLogger, TileAugentationCallback
from .loss import WeightedMSELoss, get_mse_loss, get_focal_loss, CellLoss
from .generators import get_generator


def train_patchgan(cfg, logdir):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    pyvips.cache_set_max(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: {}".format(device))

    logdir = Path(logdir)
    train_dataframe = pd.read_csv(cfg.data.train_dataframe_path)
    val_dataframe = pd.read_csv(cfg.data.val_dataframe_path)
    test_dataframe = pd.read_csv(cfg.data.test_dataframe_path)
    if cfg.data.root_dir:
        train_dataframe['image_path'] = cfg.data.root_dir + "/" + train_dataframe['image_path'].astype(str)
        val_dataframe['image_path'] = cfg.data.root_dir + "/" + val_dataframe['image_path'].astype(str)
        test_dataframe['image_path'] = cfg.data.root_dir + "/" + test_dataframe['image_path'].astype(str)
        train_dataframe['target_path'] = cfg.data.root_dir + "/" + train_dataframe['target_path'].astype(str)
        val_dataframe['target_path'] = cfg.data.root_dir + "/" + val_dataframe['target_path'].astype(str)
        test_dataframe['target_path'] = cfg.data.root_dir + "/" + test_dataframe['target_path'].astype(str)
    log.info("{} train tiles / {} val tiles / {} test tiles".format(
            len(train_dataframe), len(val_dataframe), len(test_dataframe)))
    from_slide = "image_path" not in train_dataframe.columns
    if cfg.data.slide_dataframe_path is None:
        slide_dataframe = None
    else:
        slide_dataframe = pd.read_csv(cfg.data.slide_dataframe_path)

    with open(cfg.data.channel_stats_path, "r") as f:
        channel_stats = json.load(f)

    width, height = get_width_height(train_dataframe)
    width, height = get_effective_width_height(width, height, train=True)
    nc_out = len(cfg.data.targ_channel_names)
    nc_in = 3
    log.info("{} width / {} height".format(width, height))
    log.info("{} inputs channels / {} output channels".format(nc_in, nc_out))

    channel_stats_rgb = get_new_input_mean_std(cfg, channel_stats["RGB"])
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

    channel_names = cfg.data.targ_channel_names
    # print(channel_names, channel_stats.keys())
    targ_channel_idxs = []
    for channel_name in channel_names:
        channel_info = channel_stats[channel_name]

        # 尝试多种方式获取索引
        channel_idx = None
        if "idx_channel" in channel_info:
            channel_idx = channel_info["idx_channel"]
        elif "channel_idx" in channel_info:
            channel_idx = channel_info["channel_idx"]
        else:
            # 如果都不行，尝试直接访问
            try:
                channel_idx = channel_info["idx_channel"]
            except KeyError:
                try:
                    channel_idx = channel_info["channel_idx"]
                except KeyError:
                    raise KeyError(f"在 {channel_name} 中未找到 'idx_channel' 或 'channel_idx' 键 {channel_info}")

        targ_channel_idxs.append(int(channel_idx))

    # targ_channel_idxs = [
    #     channel_stats[channel_name].get("idx_channel") or channel_stats[channel_name].get("channel_idx")
    #     for channel_name in channel_names
    # ]
    print(targ_channel_idxs,  channel_names)
    stats_list_if = [channel_stats[channel_name] for channel_name in channel_names].copy()
    preprocess_target_fn = NormalizationLayer(stats_list_if, mode="if")

    sampler_cfg = cfg.train.data_sampler
    if sampler_cfg.use_sampler:
        train_sampler = BalancedPositiveSampler(
            train_dataframe, channel_names, sampler_cfg.tresh,
            other_percent=sampler_cfg.other_percent)
    else:
        train_sampler = None

    data_module = DataModule(
        slide_dataframe=slide_dataframe, train_dataframe=train_dataframe,
        val_dataframe=val_dataframe, test_dataframe=test_dataframe,
        targ_channel_idxs=targ_channel_idxs, from_slide=from_slide,
        input_shape=(width, height),
        batch_size=cfg.train.batch_size, pin_memory=device!="cpu",
        return_nuclei=cfg.train.use_cell_metrics, train_sampler=train_sampler,
        preprocess_input_fn=preprocess_input_fn, preprocess_target_fn=preprocess_target_fn,
        )
    data_module.setup()
    train_dataloader, val_dataloader, test_dataloader = data_module.get_dataloaders()

    torch.cuda.empty_cache()

    generator = get_generator(cfg.model.model_name, width, nc_in, nc_out, cfg)

    if cfg.model.checkpoint_path:
        generator.load_state_dict(torch.load(cfg.model.checkpoint_path))
        log.info("checkpoint lodaded from {}".format(cfg.model.checkpoint_path))

    if os.name == 'nt':
        jit_compile = False
    else:
        #generator = torch.compile(generator)
        #jit_compile = True
        jit_compile = False

    ckpt_weights = str(logdir / "model.weights")

    log.info("PatchGAN training")

    if cfg.train.use_cell_metrics:
        cell_metrics = CellMetrics(slide_dataframe, channel_names)
    else:
        cell_metrics = None


    lambda_factor = cfg.train.losses.lambda_factor
    if cfg.train.losses.use_weighted_mae:
        if sampler_cfg.use_sampler:
            indices = train_sampler.create_indices()
            foreground_weight = get_foreground_weight(
                channel_names, train_sampler.dataframe.take(indices))
        else:
            foreground_weight = get_foreground_weight(
                channel_names, train_dataframe)
        foreground_weight = np.float32(foreground_weight)
        foreground_weight = torch.tensor(foreground_weight).reshape((1, -1, 1, 1)).to(device)
        foreground_thresh = preprocess_target_fn(0)

        print("foreground_weight", foreground_weight.cpu().numpy().flatten().tolist(),
            "foreground_thresh", foreground_thresh.flatten().tolist())
        #loss_reconstruct = get_weighted_mae_loss(lambda_factor, foreground_weight, foreground_thresh)
        loss_reconstruct = get_focal_loss(lambda_factor, foreground_weight)
        #loss_reconstruct = get_shrinkage_loss(lambda_factor, foreground_weight)
    else:
        #loss_reconstruct = get_mse_loss(lambda_factor)
        marker_weights = torch.Tensor([channel_stats[channel_name]["std"] \
                         for channel_name in channel_names])
        marker_weights = 1 / marker_weights
        marker_weights = marker_weights / marker_weights.min()
        print(marker_weights)
        loss_reconstruct = WeightedMSELoss(lambda_factor, marker_weights)
        #loss_reconstruct = L1_L2_Loss(lambda_factor=10.)

    cell_loss_params = cfg.train.losses.cell_loss
    if cell_loss_params.use_loss:
        cell_loss = CellLoss(
            cell_loss_params.mlp_path, nc_out, use_mse=cell_loss_params.use_mse,
            use_clustering=cell_loss_params.use_clustering, lambda_factor=lambda_factor)
    else:
        cell_loss = None

    #foreground_loss = CombinedBCEAndDiceLoss(1.)
    gan_train = cfg.train.gan_train
    selected_channels = [
        channel_stats[channel_name]["is_structural"] for channel_name in channel_names] \
            if cfg.train.gan_mode == "stuctural" else None
    discriminator = DiscriminatorPatch(
            input_nc=nc_out + nc_in, norm_layer_type=None,
            selected_channels=selected_channels) if gan_train else None

    # Adjust learning rate for DDP training (effective batch size = batch_size * num_gpus)
    use_ddp = cfg.train.get("use_ddp", False)
    num_gpus_for_lr = 1
    if use_ddp:
        num_gpus_cfg = cfg.train.get("num_gpus", None)
        if num_gpus_cfg is None:
            num_gpus_for_lr = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            num_gpus_for_lr = min(num_gpus_cfg, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    
    effective_batch_size = cfg.train.batch_size * num_gpus_for_lr
    lr_scale = np.sqrt(effective_batch_size)
    log.info(f"Effective batch size: {effective_batch_size} (batch_size: {cfg.train.batch_size}, num_gpus: {num_gpus_for_lr})")
    log.info(f"Learning rate scale: {lr_scale}")

    pl_model = ModelModule(generator=generator, discriminator=discriminator,
                     lr_g=cfg.train.learning_rate_g * lr_scale,
                     lr_d=cfg.train.learning_rate_d * lr_scale,
                     cell_metrics=cell_metrics,
                     cell_loss=cell_loss,
                     loss_reconstruct=loss_reconstruct,
                     gan_train=gan_train)

    logger_name = logdir.name
    wandb_note = cfg.train.wandb_note
    wandb_note = update_wandb_note(wandb_note)
    
    # Only initialize wandb on rank 0 for DDP
    rank = int(os.environ.get("RANK", 0))
    logger = WandbLogger(project=cfg.train.wandb_project, name=logger_name, notes=wandb_note,
                         log_model=False, save_dir=str(logdir), force=True, reinit=True)
    
    # Only save artifacts on rank 0
    if rank == 0:
        cfg_path = str(logdir / "config.yaml")
        OmegaConf.save(cfg, cfg_path)
        wandb_log_artifact(logger, "cfg", "config", cfg_path)
        wandb_log_artifact(logger, "stats_image", "stats", cfg.data.channel_stats_path)
        wandb_log_artifact(logger, "gitlog", "gitlog", str(logdir / "github_log.txt"))

    config_callback = cfg.train.callbacks
    ckpt_dirpath = str(Path(ckpt_weights).parent)
    ckpt_filename = str(Path(ckpt_weights).name)
    callbacks = [
        DebugImageLogger("logs_img", batch_frequency=1000, max_images=4, clamp=True),
        ModelCheckpoint(
            dirpath=ckpt_dirpath, filename=ckpt_filename,
            monitor=config_callback.modelcheckpoint.monitor,
            save_top_k=1,
            mode=config_callback.modelcheckpoint.mode, save_last=False,
            save_weights_only=True, verbose=1),
        WandbVisCallback(preprocess_input_fn.unormalize, num_samples=4),
        #SwitchGenDiscTrain()
    ]
    
    # Add EarlyStopping callback if enabled in config
    if hasattr(config_callback, 'early_stopping'):
        early_stop_config = config_callback.early_stopping
        try:
            if early_stop_config.enabled:
                early_stopping = EarlyStopping(
                    monitor=early_stop_config.monitor,
                    mode=early_stop_config.mode,
                    patience=int(early_stop_config.patience),
                    min_delta=float(early_stop_config.min_delta),
                    verbose=early_stop_config.verbose if hasattr(early_stop_config, 'verbose') else True,
                )
                callbacks.append(early_stopping)
                log.info(f"EarlyStopping enabled: monitor={early_stop_config.monitor}, "
                        f"mode={early_stop_config.mode}, patience={early_stop_config.patience}")
        except AttributeError:
            # If early_stopping config exists but enabled attribute is missing, skip
            pass
    if cfg.data.augmentation_dir is not None:
        if from_slide:
            callbacks.append(SlideAugentationCallback(cfg.data.augmentation_dir, prob=0.25))
        else:
            callbacks.append(TileAugentationCallback(cfg.data.augmentation_dir, prob=0.25))

    pl_model = pl_model.to(device)
    if jit_compile:
        pl_model = torch.compile(pl_model)

    # Configure DDP settings from config
    use_ddp = cfg.train.get("use_ddp", False)
    num_gpus = cfg.train.get("num_gpus", None)
    strategy = cfg.train.get("strategy", "ddp")
    
    # Check if we're in a distributed environment (launched by torchrun)
    is_distributed = (
        "RANK" in os.environ or 
        "LOCAL_RANK" in os.environ or 
        "WORLD_SIZE" in os.environ
    )

    trainer_kwargs = dict(
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu",
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
    )
    max_steps = cfg.train.get("max_steps", None)
    if max_steps is not None:
        trainer_kwargs["max_steps"] = int(max_steps)
    else:
        trainer_kwargs["max_epochs"] = cfg.train.epochs

    eval_every_n_steps = cfg.train.get("eval_every_n_steps", None)
    if eval_every_n_steps is not None:
        trainer_kwargs["val_check_interval"] = int(eval_every_n_steps)

    log_every_n_steps = cfg.train.get("log_every_n_steps", None)
    if log_every_n_steps is not None:
        trainer_kwargs["log_every_n_steps"] = int(log_every_n_steps)
    
    if use_ddp:
        # Determine number of GPUs
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            num_gpus = min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        
        if num_gpus > 1:
            # Use DDP strategy
            if strategy == "ddp" or strategy == "auto":
                ddp_strategy = "ddp"
            elif strategy == "ddp_find_unused_parameters_false":
                ddp_strategy = "ddp_find_unused_parameters_false"
            else:
                ddp_strategy = strategy
            
            if is_distributed:
                # Already launched by torchrun, use the distributed context
                log.info(f"DDP training detected: RANK={os.environ.get('RANK', 'N/A')}, "
                        f"WORLD_SIZE={os.environ.get('WORLD_SIZE', 'N/A')}, strategy: {ddp_strategy}")
                # When launched by torchrun, PyTorch Lightning will use the distributed context
                # We need to set devices to match the world size
                world_size = int(os.environ.get("WORLD_SIZE", num_gpus))
                trainer = Trainer(
                    **trainer_kwargs,
                    devices=world_size,
                    strategy=ddp_strategy,
                )#limit_train_batches=100, limit_val_batches=100, limit_test_batches=100)
            else:
                # Not in distributed context, PyTorch Lightning will auto-launch
                log.info(f"Using DDP training with {num_gpus} GPUs, strategy: {ddp_strategy}")
                trainer = Trainer(
                    **trainer_kwargs,
                    devices=num_gpus,
                    strategy=ddp_strategy,
                )#limit_train_batches=100, limit_val_batches=100, limit_test_batches=100)
        else:
            log.info("DDP requested but only 1 GPU available, using single GPU training")
            trainer = Trainer(
                **trainer_kwargs,
                devices=1,
            )#limit_train_batches=100, limit_val_batches=100, limit_test_batches=100)
    else:
        trainer = Trainer(
            **trainer_kwargs,
            devices=1,
        )#limit_train_batches=100, limit_val_batches=100, limit_test_batches=100)
    trainer.fit(pl_model, train_dataloader, val_dataloader)
    trainer.test(pl_model, test_dataloader, ckpt_path=ckpt_weights + ".ckpt", verbose=True)
    
    # Only finish wandb on rank 0 (main script will also call wandb.finish())
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        wandb.finish()
