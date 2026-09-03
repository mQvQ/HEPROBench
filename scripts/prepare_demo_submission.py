from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from heprobench.config import load_dataset_config
from heprobench.io_h5 import write_slide_h5
from heprobench.schema import group_by_slide, load_channel_names, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic valid/test predictions for the evaluation demo")
    parser.add_argument("--config", default="configs/demo/rosie.json")
    parser.add_argument("--output", default="outputs/evaluation_demo")
    args = parser.parse_args()

    output_root = Path(args.output).expanduser().resolve()
    rng = np.random.default_rng(2026)
    for split in ("valid", "test"):
        cfg, _ = load_dataset_config(args.config, split_override=split)
        dataset = cfg["dataset"]
        channels = load_channel_names(dataset["channel_names"])
        for slide_name, records in group_by_slide(load_records(dataset)).items():
            predictions = []
            for record in records:
                if record.target_path is None:
                    raise ValueError(f"Demo target missing: {record}")
                target = np.load(record.target_path).astype(np.int16)
                noise = rng.integers(-8, 9, size=target.shape, dtype=np.int16)
                predictions.append(np.clip(target + noise, 0, 255).astype(np.uint8))
            write_slide_h5(
                output_root / split / f"{slide_name}.h5",
                np.stack(predictions),
                records,
                channels,
                {
                    "method_name": "synthetic-evaluation-smoke-test",
                    "split": split,
                    "patch_size": int(dataset.get("patch_size", 256)),
                    "demo_only": True,
                },
            )
    print(f"Prepared valid/test evaluation submission under {output_root}")


if __name__ == "__main__":
    main()
