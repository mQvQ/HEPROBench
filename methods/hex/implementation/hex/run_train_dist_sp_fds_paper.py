"""
Runner that "completes" the upstream HEX training script without editing it:
- supplies missing symbols (seed_torch/PatchDataset/print_network)
- swaps in a CustomModel variant that includes FDS

Usage:
  torchrun --nnodes=1 --nproc-per-node=8 benchmark/methods/HEX/hex/run_train_dist_codex_lung_marker_fds.py
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]  # benchmark/methods/HEX
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hex.train_dist_sp_fds_paper import main


if __name__ == "__main__":
    main()
