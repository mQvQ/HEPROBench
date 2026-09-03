from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from os.path import join
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import robust_loss_pytorch

from hex.fds import FDSConfig
from hex.hex_architecture_fds import CustomModelFDS
from hex.utils import CustomDataset, print_network, seed_torch


def setup_distributed() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True, help="Root containing *_patch_meta.csv and images/targets")
    p.add_argument("--save_dir", type=str, required=True, help="Output directory for runs/ and checkpoints/")
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--eval_interval", type=int, default=10_000)
    p.add_argument("--ckpt_interval", type=int, default=10_000)
    p.add_argument("--stage1_iters", type=int, default=83_333, help="Train last 4 encoder layers + heads")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_gamma", type=float, default=0.95, help="Exponential decay factor per epoch")
    p.add_argument("--batch_size_per_gpu", type=int, default=16, help="Benchmark setting (per-process batch size)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--musk_img_size", type=int, default=384, help="Must match MUSK config, e.g. 256 or 384")
    p.add_argument("--pretrained_ckpt", type=str, default="", help="Optional checkpoint to load (state_dict)")
    p.add_argument("--panel_key", type=str, default="default")
    p.add_argument("--norm", choices=["paper", "imagenet"], default="imagenet")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--log_file", type=str, default="", help="Default: <save_dir>/train.log")

    p.add_argument("--label_scale", type=float, default=255.0, help="Divide targets by this value (default: 255 -> [0,1])")

    p.add_argument("--fds_bins", type=int, default=100)
    p.add_argument("--fds_label_min", type=float, default=0.0)
    p.add_argument("--fds_label_max", type=float, default=1.0)
    p.add_argument("--fds_start_update", type=int, default=0)
    p.add_argument("--fds_start_apply", type=int, default=10)
    p.add_argument("--distributed", action="store_true", help="Enable DDP (torchrun). Default is single-GPU.")
    return p.parse_args()


def _set_trainable(model: CustomModelFDS, *, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if stage == "stage1":
        for layer in model.visual.beit3.encoder.layers[-4:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in model.visual.beit3.encoder.layer_norm.parameters():
            p.requires_grad = True
    elif stage == "stage2":
        pass
    else:
        raise ValueError(f"unknown stage: {stage}")

    for p in model.regression_head.parameters():
        p.requires_grad = True
    for p in model.regression_head1.parameters():
        p.requires_grad = True


def _all_reduce_moments(moments: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count, sum1, sum2 = moments
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum1, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum2, op=dist.ReduceOp.SUM)
    return count, sum1, sum2


def main() -> None:
    args = parse_args()
    if args.distributed:
        setup_distributed()
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        seed_torch(global_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed_torch(0)

    if args.label_scale <= 0:
        raise ValueError("--label_scale must be > 0")

    save_dir = args.save_dir
    dataroot = args.dataroot
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = args.log_file or join(save_dir, "train.log")
    log_f = open(log_path, "a", encoding="utf-8") if global_rank == 0 else None

    def _log(message: str) -> None:
        if global_rank != 0:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        if log_f is not None:
            log_f.write(line + "\n")
            log_f.flush()

    _log("Starting HEX training (targets normalized to [0,1])")
    _log("Args: " + json.dumps(vars(args), sort_keys=True))

    if args.norm == "paper":
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

        normalize = transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    transform_train = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = CustomDataset(dataroot=dataroot, phase="train", panel_key=args.panel_key, transform=transform_train)
    val_dataset = CustomDataset(dataroot=dataroot, phase="valid", panel_key=args.panel_key, transform=transform_val)
    num_outputs = int(np.asarray(train_dataset[0][1]).shape[0])

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        train_shuffle = False
        val_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=val_sampler,
        shuffle=val_shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    fds_config = FDSConfig(
        num_bins=args.fds_bins,
        label_min=args.fds_label_min,
        label_max=args.fds_label_max,
        feature_dim=128,
        start_update=args.fds_start_update,
        start_apply=args.fds_start_apply,
    )

    model = CustomModelFDS(
        visual_output_dim=1024,
        num_outputs=num_outputs,
        fds_config=fds_config,
        musk_img_size=args.musk_img_size,
    ).to(device)

    if args.pretrained_ckpt:
        state_dict = torch.load(args.pretrained_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    model_for_cfg = model.module if args.distributed else model
    _set_trainable(model_for_cfg, stage="stage1")

    if global_rank == 0:
        print_network(model)
        _log("Stage: stage1 (last 4 encoder layers + heads trainable)")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    criterion_ad = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=num_outputs, float_dtype=torch.float32, device=local_rank)
    optimizer.add_param_group({"params": criterion_ad.parameters(), "lr": args.lr, "name": "criterion_ad"})

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    global_step = 0
    epoch = 0
    while global_step < args.max_iters:
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if global_step >= args.stage1_iters:
            _set_trainable(model_for_cfg, stage="stage2")
            if global_rank == 0:
                _log(f"Stage switch: stage2 at step={global_step} epoch={epoch} (heads only trainable)")
        else:
            _set_trainable(model_for_cfg, stage="stage1")

        model.train()
        model_for_cfg.training_status = True

        moments = model_for_cfg.FDS.init_moments(device=device)
        running_loss = 0.0
        step_in_epoch = 0

        for inputs, labels, *_ in train_loader:
            if global_step >= args.max_iters:
                break

            inputs = inputs.to(device, non_blocking=True, dtype=torch.float16)
            labels = labels.to(device, non_blocking=True, dtype=torch.float16)
            if args.label_scale != 1.0:
                labels = labels / float(args.label_scale)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                outputs, raw_features = model(inputs, labels, epoch)
                loss = torch.mean(criterion_ad.lossfun(outputs.to(dtype=torch.float32) - labels.to(dtype=torch.float32)))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            step_in_epoch += 1
            global_step += 1

            moments = model_for_cfg.FDS.accumulate_moments(
                moments,
                raw_features.detach().to(dtype=torch.float32),
                labels.detach().to(dtype=torch.float32),
            )

            if global_step % args.log_interval == 0:
                lr0 = optimizer.param_groups[0]["lr"]
                _log(f"step={global_step} epoch={epoch} loss={float(loss.item()):.6f} lr={lr0:.2e}")

            if global_step % args.eval_interval == 0:
                model.eval()
                model_for_cfg.training_status = False

                all_labels = []
                all_preds = []
                autocast_eval_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                with torch.no_grad(), autocast_eval_ctx:
                    for v_inputs, v_labels, *_ in val_loader:
                        v_inputs = v_inputs.to(device, non_blocking=True, dtype=torch.float16)
                        v_labels = v_labels.to(device, non_blocking=True, dtype=torch.float16)
                        if args.label_scale != 1.0:
                            v_labels = v_labels / float(args.label_scale)
                        v_outputs, _ = model(v_inputs, v_labels, epoch)
                        all_labels.append(v_labels.to(dtype=torch.float32))
                        all_preds.append(v_outputs.to(dtype=torch.float32))

                all_labels_t = torch.cat(all_labels, dim=0)
                all_preds_t = torch.cat(all_preds, dim=0)

                if args.distributed:
                    gathered_labels = [torch.zeros_like(all_labels_t) for _ in range(world_size)]
                    gathered_preds = [torch.zeros_like(all_preds_t) for _ in range(world_size)]
                    dist.all_gather(gathered_labels, all_labels_t)
                    dist.all_gather(gathered_preds, all_preds_t)
                    if global_rank == 0:
                        labels_np = torch.cat(gathered_labels).cpu().numpy()
                        preds_np = torch.cat(gathered_preds).cpu().numpy()
                        mse_per_marker = np.nanmean((labels_np - preds_np) ** 2, axis=0)
                        overall_mse = float(np.nanmean(mse_per_marker))
                        _log(f"eval step={global_step} epoch={epoch} mse_val_avg={overall_mse:.6f}")
                else:
                    labels_np = all_labels_t.cpu().numpy()
                    preds_np = all_preds_t.cpu().numpy()
                    mse_per_marker = np.nanmean((labels_np - preds_np) ** 2, axis=0)
                    overall_mse = float(np.nanmean(mse_per_marker))
                    _log(f"eval step={global_step} epoch={epoch} mse_val_avg={overall_mse:.6f}")

                model.train()
                model_for_cfg.training_status = True

            if global_rank == 0 and global_step % args.ckpt_interval == 0:
                state = model.module.state_dict() if args.distributed else model.state_dict()
                ckpt_path = join(checkpoint_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(state, ckpt_path)
                _log(f"saved checkpoint: {ckpt_path}")

        avg_loss = torch.tensor(running_loss / max(1, step_in_epoch), device=device, dtype=torch.float32)
        if args.distributed:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / world_size
        _log(f"epoch_end epoch={epoch} step={global_step} loss_avg={float(avg_loss.item()):.6f}")

        if epoch >= model_for_cfg.FDS.start_update:
            if args.distributed:
                moments = _all_reduce_moments(moments)
            model_for_cfg.FDS.update_running_stats_from_moments(*moments, epoch=epoch)
            model_for_cfg.FDS.update_last_epoch_stats(epoch)
            _log(f"fds_update epoch={epoch} stats_ready={bool(model_for_cfg.FDS.stats_ready.item())}")

        scheduler.step()
        if args.distributed:
            dist.barrier()
        epoch += 1

    if log_f is not None:
        log_f.close()
    cleanup()


if __name__ == "__main__":
    main()

