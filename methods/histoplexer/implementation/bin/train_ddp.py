import argparse
import json
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add the project root directory to Python path
# This allows importing src modules when running as python -m bin.train_ddp
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset.dataset import TuProDataset
from src.trainers.histoplexer_trainer import HistoplexerTrainer
from src.config.config import Config
from src.utils.misc import seed_everything


def setup_distributed(rank, world_size, backend='nccl', init_method='env://', master_addr=None, master_port=None, device_id=None):
    """Initialize the distributed training environment."""
    # Only set environment variables if not already set (e.g., by torchrun)
    if init_method == 'env://':
        if 'MASTER_ADDR' not in os.environ:
            if master_addr is not None:
                os.environ['MASTER_ADDR'] = master_addr
            else:
                os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            if master_port is not None:
                os.environ['MASTER_PORT'] = str(master_port)
            else:
                # Find an available port starting from 12345
                import socket
                port = 12345
                while port < 12400:  # Try ports 12345-12399
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('', port))
                        os.environ['MASTER_PORT'] = str(port)
                        break
                    except OSError:
                        port += 1
                else:
                    raise RuntimeError("Could not find an available port in range 12345-12399")

    # Set CUDA device before initializing process group (critical for NCCL)
    # NCCL requires the device to be set before init_process_group
    if device_id is not None:
        torch.cuda.set_device(device_id)
        # For NCCL backend, create a device object to pass to init_process_group
        # This ensures NCCL knows which device to use
        device = torch.device(f'cuda:{device_id}')
    else:
        device = None

    # initialize the process group
    # For NCCL, the device must be set via torch.cuda.set_device() before this call
    # PyTorch will automatically use the current device for NCCL
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)


def cleanup_distributed():
    """Clean up the distributed training environment."""
    dist.destroy_process_group()


def main_worker(rank, world_size, config_path):
    """Main worker function for distributed training."""
    # Setup distributed training
    if isinstance(config_path, str) and config_path.endswith('.json'):
        with open(config_path, "r") as ifile:
            config = Config(json.load(ifile))
    elif isinstance(config_path, str):
        # Assume it's a path to a config file without .json extension
        with open(config_path, "r") as ifile:
            config = Config(json.load(ifile))
    else:
        # Assume it's already a Config object
        config = config_path

    # Override device for DDP - each process gets its own GPU
    # Use LOCAL_RANK if available (set by torchrun), otherwise use rank
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    config.device = f'cuda:{local_rank}'

    # Setup distributed environment (device_id must be set before init_process_group for NCCL)
    setup_distributed(rank, world_size, config.ddp_backend, config.ddp_init_method, 
                      master_addr=config.ddp_master_addr, master_port=config.ddp_master_port,
                      device_id=local_rank)

    # Set the device for this process (already set in setup_distributed, but ensure it's set)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Only rank 0 prints experiment name and creates directories
    if rank == 0:
        # experiment name
        if config.resume_path == None:
            channel_str = 'all' if config.channels is None else str(config.markers[config.channels[0]])
            config.experiment_name  = config.cohort + '_' + config.method + '_channels-' + channel_str  +'_seed-' + str(config.seed)
        else:
            config.experiment_name = config.resume_path.split('/')[-1]

        print(config.experiment_name)

        config.save_path = os.path.join(config.base_save_path, config.experiment_name)
        print(f"save path: {config.save_path}")
        # create output folder if it doesn't exist yet
        Path(config.save_path).mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved in {config.save_path}")
        print(f"Using DDP with world_size: {world_size}")

        # save config file
        with open(os.path.join(config.save_path, "config.json"), "w") as ofile:
            json.dump(config.__dict__, ofile, indent=4)
    else:
        config.experiment_name = "ddp_training"
        config.save_path = os.path.join(config.base_save_path, config.experiment_name)

    # Seed everything
    seed_everything(seed=config.seed, device=device)

    # Create datasets
    train_dataset = TuProDataset(
        split=config.split,
        mode='train',
        src_folder=config.src_folder,
        tgt_folder=config.tgt_folder,
        use_high_res=config.use_high_res,
        p_flip_jitter_hed_affine=config.p_flip_jitter_hed_affine,
        patch_size=config.patch_size,
        channels=config.channels,
        cohort=config.cohort,
        use_fm_features=config.use_fm_features,
        fm_features_path=config.fm_features_path
        )

    datasets = [train_dataset]

    if config.val:
        val_dataset = TuProDataset(
            split=config.split,
            mode='valid',
            src_folder=config.src_folder,
            tgt_folder=config.tgt_folder,
            use_high_res=config.use_high_res,
            p_flip_jitter_hed_affine=config.p_flip_jitter_hed_affine,
            patch_size=config.patch_size,
            channels=config.channels
        )
        datasets.append(val_dataset)

    if rank == 0:
        print(f"Number of training images: {len(train_dataset)}")

    # initialize trainer with DDP support
    trainer = HistoplexerTrainer(args=config, datasets=datasets, rank=rank, world_size=world_size)
    trainer.train()

    # Clean up
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Configurations for HistoPlexer DDP Training")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument("--master_port", type=str, default=None, help="Master port for DDP (overrides config file)")
    args = parser.parse_args()

    # Check if running under torchrun
    rank = os.environ.get('RANK')
    local_rank = os.environ.get('LOCAL_RANK')
    world_size = os.environ.get('WORLD_SIZE')

    if rank is None or local_rank is None or world_size is None:
        print("Error: This script must be launched using torchrun.")
        print("Usage: torchrun --nproc_per_node=<num_gpus> -m bin.train_ddp --config_path=<config_path>")
        print("Or use the provided script: bash train_ddp.sh <config_path> <num_gpus> [master_port]")
        sys.exit(1)

    # Convert to integers
    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    # load config file
    with open(args.config_path, "r") as ifile:
        config = Config(json.load(ifile))

    # Override master port if provided via command line
    if args.master_port is not None:
        config.ddp_master_port = args.master_port
        # Also set environment variable if not already set
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = args.master_port

    # Check if DDP is enabled
    if not config.ddp_enabled:
        print("DDP is not enabled in config. Use train.py for single GPU training.")
        return

    # Verify world_size matches config
    if world_size != config.ddp_world_size:
        if rank == 0:
            print(f"Warning: WORLD_SIZE ({world_size}) does not match config.ddp_world_size ({config.ddp_world_size})")
            print(f"Using WORLD_SIZE={world_size} from torchrun")

    # Call main_worker directly (torchrun handles process spawning)
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
