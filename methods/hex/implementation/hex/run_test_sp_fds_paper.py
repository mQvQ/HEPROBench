"""
Runner that supplies missing symbols and loads the FDS-enabled model class.

Note: if you load the provided `checkpoint.pth` from the upstream repo, it will
not contain FDS running/smoothed statistics. The model will still load with
`strict=False`, but feature calibration will remain disabled unless you train
and save a checkpoint that includes the `.FDS` buffers.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]  # benchmark/methods/HEX
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hex.test_customdataset_hex_fds import main


if __name__ == "__main__":
    main()
