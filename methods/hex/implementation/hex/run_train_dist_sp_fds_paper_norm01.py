"""
Runner that uses the 0-1 target-normalized HEX training script.

Usage (example):
  torchrun --nnodes=1 --nproc-per-node=8 benchmark/methods/HEX/hex/run_train_dist_sp_fds_paper_norm01.py \
    --distributed \
    --dataroot /path/to/preprocess/data/multi-tumor-codex-preprocess/multi-tumor-codex-reg-patches \
    --save_dir /path/to/benchmark/results/HEX/mt-codex-norm01
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]  # benchmark/methods/HEX
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hex.train_dist_sp_fds_paper_norm01 import main


if __name__ == "__main__":
    main()

