from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.hepro_pipeline import run_hepro
from methods.runner_utils import load_config, parser, run_dir


def main() -> int:
    args = parser("HEPROBench adapter for MIPHEI-ViT").parse_args()
    config, source = load_config(args.config)
    return run_hepro(
        config=config,
        source=source,
        destination=run_dir(config, args.task),
        task=args.task,
        variant="miphei_vit",
        encoder=None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
