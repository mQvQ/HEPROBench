# HistoPlexer DDP Training Guide

This guide explains how to use Distributed Data Parallel (DDP) training with HistoPlexer to train larger batch sizes across multiple GPUs.

## Prerequisites

- PyTorch >= 1.9.0 (recommended >= 1.10.0 for torchrun)
- Multiple GPUs available
- CUDA compatible with your PyTorch version

## Configuration

### DDP Configuration

Create a configuration file with DDP settings:

```json
{
    "src_folder": "/path/to/data",
    "tgt_folder": "/path/to/data",
    "split": "/path/to/split.csv",
    "markers": ["CD16", "CD20", "CD3", "CD31", "CD8a"],
    "cohort": "your_cohort",
    "use_gp": true,
    "w_GP": 5.0,
    "w_ASP": 1.0,
    "batch_size": 64,  // Increased batch size for multi-GPU
    "output_nc": 5,
    "device": "cuda",
    "method": "ours_ddp",
    "base_save_path": "/path/to/save",
    "ddp": {
        "enabled": true,
        "world_size": 4,
        "backend": "nccl",
        "init_method": "env://",
        "master_addr": "localhost",
        "master_port": "12345"
    }
}
```

**Key points for DDP config:**
- Set `"ddp.enabled": true`
- Set `"world_size"` to the number of GPUs you want to use
- Increase `batch_size` proportionally to the number of GPUs
- Set `"device": "cuda"` (DDP will automatically assign specific GPUs to each process)

### Single GPU Configuration

For single GPU training, use:

```json
{
    "ddp": {
        "enabled": false
    },
    "device": "cuda:0"
}
```

## Training Scripts

### Option 1: Using torchrun (Recommended)

```bash
# Navigate to the HistoPlexer directory
cd /path/to/HistoPlexer

# Run DDP training with 4 GPUs
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    -m bin.train_ddp \
    --config_path=src/config/your_config_ddp.json
```

### Option 2: Using the provided script

```bash
# Make the script executable
chmod +x train_ddp_simple.sh

# Run DDP training
./train_ddp_simple.sh src/config/your_config_ddp.json 4
```

### Option 3: Using the legacy launch script (now uses torchrun)

```bash
# Make the script executable
chmod +x train_ddp.sh

# Run DDP training
./train_ddp.sh src/config/your_config_ddp.json 4
```

### IMC 0–1 variant

To run the IMC-scaled (0–1) experiment, use the matching script:

```bash
./train_ddp_imc01.sh src/config/your_config_ddp.json 4
```

## Single GPU Training

Single GPU training continues to work as before:

```bash
python bin/train.py --config_path=src/config/your_config_single.json
```

## Troubleshooting

### Common Issues

1. **"can't open file 'bin.train_ddp'"**
   - Use `bin/train_ddp.py` instead of `bin.train_ddp`
   - Ensure you're in the correct directory

2. **"NCCL error: remote process exited or there was a network error"**
   - This has been addressed with process synchronization barriers
   - Check that all GPUs have sufficient memory available
   - Ensure no other processes are using the GPUs
   - The error messages now include detailed memory usage information

3. **CUDA out of memory**
   - Reduce batch_size in your config
   - Use fewer GPUs

4. **Port already in use**
   - Change the master_port in your config or script

5. **torch.distributed.launch is deprecated**
   - Both scripts now use `torchrun` (recommended for PyTorch >= 1.10)

### Environment Variables

You can also set DDP parameters via environment variables:

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0

torchrun bin/train_ddp.py --config_path=your_config.json
```

## Batch Size Scaling

When using N GPUs with DDP, you can typically use N times larger batch size compared to single GPU training. For example:

- Single GPU: batch_size = 16
- 4 GPUs: batch_size = 64

This is because the total batch size across all GPUs is `batch_size * world_size`.

## Monitoring Training

- Training logs are only shown for rank 0 process
- Checkpoints are saved by rank 0 process
- Use `nvidia-smi` to monitor GPU usage across all processes
