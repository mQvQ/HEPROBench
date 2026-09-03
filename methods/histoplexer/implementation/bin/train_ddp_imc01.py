import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.config import Config
from src.dataset.dataset_imc01 import TuProDatasetIMC01
from src.trainers.histoplexer_trainer import HistoplexerTrainer
from src.utils.misc import seed_everything


def _ensure_imc01_method(config: Config) -> None:
    if not str(config.method).endswith("_imc01"):
        config.method = f"{config.method}_imc01"


def setup_distributed(
    rank, world_size, backend="nccl", init_method="env://", master_addr=None, master_port=None, device_id=None
):
    if init_method == "env://":
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_addr or "localhost"
        if "MASTER_PORT" not in os.environ:
            if master_port is not None:
                os.environ["MASTER_PORT"] = str(master_port)
            else:
                import socket

                port = 12345
                while port < 12400:
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(("", port))
                        os.environ["MASTER_PORT"] = str(port)
                        break
                    except OSError:
                        port += 1
                else:
                    raise RuntimeError("Could not find an available port in range 12345-12399")

    if device_id is not None:
        torch.cuda.set_device(device_id)

    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)


def cleanup_distributed():
    dist.destroy_process_group()


def main_worker(rank, world_size, config_path):
    if isinstance(config_path, str) and config_path.endswith(".json"):
        with open(config_path, "r") as ifile:
            config = Config(json.load(ifile))
    elif isinstance(config_path, str):
        with open(config_path, "r") as ifile:
            config = Config(json.load(ifile))
    else:
        config = config_path

    _ensure_imc01_method(config)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    config.device = f"cuda:{local_rank}"

    setup_distributed(
        rank,
        world_size,
        config.ddp_backend,
        config.ddp_init_method,
        master_addr=config.ddp_master_addr,
        master_port=config.ddp_master_port,
        device_id=local_rank,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        if config.resume_path is None:
            channel_str = "all" if config.channels is None else str(config.markers[config.channels[0]])
            config.experiment_name = f"{config.cohort}_{config.method}_channels-{channel_str}_seed-{config.seed}"
        else:
            config.experiment_name = config.resume_path.split("/")[-1]

        print(config.experiment_name)
        config.save_path = os.path.join(config.base_save_path, config.experiment_name)
        print(f"save path: {config.save_path}")
        Path(config.save_path).mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved in {config.save_path}")
        print(f"Using DDP with world_size: {world_size}")

        with open(os.path.join(config.save_path, "config.json"), "w") as ofile:
            json.dump(config.__dict__, ofile, indent=4)
    else:
        config.experiment_name = "ddp_training"
        config.save_path = os.path.join(config.base_save_path, config.experiment_name)

    seed_everything(seed=config.seed, device=device)

    train_dataset = TuProDatasetIMC01(
        split=config.train_csv or config.split,
        mode="train",
        src_folder=config.src_folder,
        tgt_folder=config.tgt_folder,
        use_high_res=config.use_high_res,
        p_flip_jitter_hed_affine=config.p_flip_jitter_hed_affine,
        patch_size=config.patch_size,
        channels=config.channels,
        cohort=config.cohort,
        use_fm_features=config.use_fm_features,
        fm_features_path=config.fm_features_path,
    )

    datasets = [train_dataset]

    if config.val:
        val_dataset = TuProDatasetIMC01(
            split=config.val_csv or config.split,
            mode="valid",
            src_folder=config.src_folder,
            tgt_folder=config.tgt_folder,
            use_high_res=config.use_high_res,
            p_flip_jitter_hed_affine=config.p_flip_jitter_hed_affine,
            patch_size=config.patch_size,
            channels=config.channels,
            cohort=config.cohort,
            use_fm_features=config.use_fm_features,
            fm_features_path=config.fm_features_path,
        )
        datasets.append(val_dataset)

    if rank == 0:
        print(f"Number of training images: {len(train_dataset)}")

    trainer = HistoplexerTrainer(args=config, datasets=datasets, rank=rank, world_size=world_size)
    trainer.train()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="HistoPlexer DDP Training (IMC scaled to 0-1)")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument("--master_port", type=str, default=None, help="Master port for DDP (overrides config file)")
    args = parser.parse_args()

    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")

    if rank is None or local_rank is None or world_size is None:
        print("Error: This script must be launched using torchrun.")
        print("Usage: torchrun --nproc_per_node=<num_gpus> -m bin.train_ddp_imc01 --config_path=<config_path>")
        sys.exit(1)

    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    with open(args.config_path, "r") as ifile:
        config = Config(json.load(ifile))

    _ensure_imc01_method(config)

    if args.master_port is not None:
        config.ddp_master_port = args.master_port
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = args.master_port

    if not config.ddp_enabled:
        print("DDP is not enabled in config. Use train_imc01.py for single GPU training.")
        return

    if world_size != config.ddp_world_size and rank == 0:
        print(f"Warning: WORLD_SIZE ({world_size}) != config.ddp_world_size ({config.ddp_world_size})")

    try:
        main_worker(rank, world_size, config)
    except KeyboardInterrupt:
        if rank == 0:
            print("Training interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
