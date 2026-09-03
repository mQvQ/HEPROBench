from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.hepro_pipeline import run_hepro
from methods.runner_utils import load_config, object_at, parser, run_dir


def main() -> int:
    args = parser("HEPROBench adapter for DPT with a frozen pathology foundation model").parse_args()
    config, source = load_config(args.config)
    encoder_cfg = object_at(config, "model", "encoder")
    encoder = str(encoder_cfg.get("name") or encoder_cfg.get("encoder_name") or "")
    if not encoder:
        raise ValueError("DPT+FM requires model.encoder.name")
    return run_hepro(
        config=config,
        source=source,
        destination=run_dir(config, args.task),
        task=args.task,
        variant="dpt_fm",
        encoder=encoder,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
