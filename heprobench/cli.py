from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dispatch import dispatch_method_task
from .registry import load_encoder_registry, load_method_registry


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _add_unified_run_arguments(parser: argparse.ArgumentParser, *, task: str) -> None:
    parser.add_argument("--encoder", default=None, help="Foundation-model encoder override (dpt_fm only)")
    parser.add_argument("--split", default=None, help="Dataset split override")
    parser.add_argument("--device", default=None, help="Runtime device override, e.g. cuda:0")
    parser.add_argument("--batch-size", type=int, default=None, help=f"{task} batch-size override")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set an arbitrary JSON value using a dotted key; may be repeated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print the command without running it")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HEPROBench unified benchmark CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-methods", help="List registered benchmark methods and supported tasks")
    sub.add_parser("list-encoders", help="List the 15 registered pathology foundation models")

    p = sub.add_parser("validate-data", help="Validate dataset metadata and target arrays")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default=None, help="Dataset split override")
    p.add_argument("--skip-arrays", action="store_true")

    p = sub.add_parser("infer", help="Run a method config and write HDF5 submission files")
    p.add_argument("--config", required=True)
    p.add_argument("--method", default=None, help="Optional JSON method override, or required legacy demo method config")
    p.add_argument("--output", "--output-dir", dest="output_dir", default=None)
    _add_unified_run_arguments(p, task="inference")

    p = sub.add_parser("train", help="Run the registered method-specific training pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--method", default=None, help="Optional JSON method override, or required legacy demo method config")
    p.add_argument("--output-dir", default=None, help="Unified run directory override")
    p.add_argument("--output-checkpoint", default=None, help="Legacy demo checkpoint output")
    p.add_argument("--epochs", type=int, default=None, help="Legacy demo epoch count")
    p.add_argument("--lr", type=float, default=None, help="Legacy demo learning rate")
    _add_unified_run_arguments(p, task="training")

    p = sub.add_parser("validate-submission", help="Validate HDF5 submission files")
    p.add_argument("--config", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--split", default=None, help="Dataset split override")

    p = sub.add_parser("evaluate", help="Evaluate a HDF5 submission against demo targets")
    p.add_argument("--config", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--output-csv", default=None)
    p.add_argument(
        "--valid-pred-dir",
        default=None,
        help="Validation predictions used to train the cell-level XGBoost classifiers",
    )
    p.add_argument("--cells", dest="cells", action="store_true", help="Enable cell-level evaluation")
    p.add_argument("--no-cells", dest="cells", action="store_false", help="Disable cell-level evaluation")
    p.add_argument(
        "--perceptual",
        dest="perceptual",
        action="store_true",
        help="Enable the configured LPIPS/DISTS evaluation",
    )
    p.add_argument(
        "--no-perceptual",
        dest="perceptual",
        action="store_false",
        help="Skip LPIPS/DISTS while retaining RMSE/PSNR/SSIM and cell metrics",
    )
    p.add_argument("--efficiency-json", default=None, help="Result produced by `heprobench profile`")
    p.set_defaults(cells=None, perceptual=None)

    p = sub.add_parser("profile", help="Measure native model parameters, FLOPs, and inference latency")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("preprocess", help="Run the reproducible dataset preprocessing/QC pipeline")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Stage to run; repeat for multiple stages. Choices: all, split, register, "
            "registration-qc, tile, normalize, patch-qc, segment, cell-extract, gate"
        ),
    )
    p.add_argument("--output-dir", default=None, help="Preprocessing output-root override")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an arbitrary JSON value using a dotted key; may be repeated",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve and print the stage plan without writing files")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-methods":
        methods = load_method_registry()
        _print_json(
            {
                name: {
                    "display_name": spec.display_name,
                    "paradigm": spec.paradigm,
                    "tasks": sorted(spec.tasks),
                }
                for name, spec in methods.items()
            }
        )
    elif args.command == "list-encoders":
        encoders = load_encoder_registry()
        _print_json(
            {
                name: {
                    "display_name": spec.display_name,
                    "family": spec.family,
                    "source": spec.source,
                    "gated": spec.gated,
                    "checkpoint": spec.checkpoint,
                    "benchmark_input_size": list(spec.benchmark_input_size),
                    "normalization": spec.normalization,
                    "feature_layers": list(spec.feature_layers),
                }
                for name, spec in encoders.items()
            }
        )
    elif args.command == "validate-data":
        from .validate import validate_data

        _print_json(
            validate_data(
                args.config,
                check_arrays=not args.skip_arrays,
                split_override=args.split,
            )
        )
    elif args.command == "infer":
        if Path(args.config).suffix.lower() == ".json":
            _print_json(
                dispatch_method_task(
                    task="infer",
                    method_name=args.method,
                    config_path=args.config,
                    encoder=args.encoder,
                    split=args.split,
                    device=args.device,
                    batch_size=args.batch_size,
                    output_dir=args.output_dir,
                    overrides=args.set,
                    dry_run=args.dry_run,
                )
            )
        else:
            from .infer import run_inference

            if not args.method:
                parser.error("legacy YAML inference requires --method")
            if not args.output_dir:
                parser.error("legacy YAML inference requires --output")
            _print_json(
                run_inference(
                    args.config,
                    args.method,
                    Path(args.output_dir),
                    device=args.device or "cpu",
                )
            )
    elif args.command == "train":
        if Path(args.config).suffix.lower() == ".json":
            _print_json(
                dispatch_method_task(
                    task="train",
                    method_name=args.method,
                    config_path=args.config,
                    encoder=args.encoder,
                    split=args.split,
                    device=args.device,
                    batch_size=args.batch_size,
                    output_dir=args.output_dir,
                    overrides=args.set,
                    dry_run=args.dry_run,
                )
            )
        else:
            from .train import run_training

            if not args.method:
                parser.error("legacy YAML training requires --method")
            if not args.output_checkpoint:
                parser.error("legacy YAML training requires --output-checkpoint")
            _print_json(
                run_training(
                    args.config,
                    args.method,
                    args.output_checkpoint,
                    epochs=args.epochs or 1,
                    lr=args.lr or 1e-3,
                    device=args.device or "cpu",
                )
            )
    elif args.command == "validate-submission":
        from .validate import validate_submission

        _print_json(validate_submission(args.config, args.pred_dir, split_override=args.split))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        _print_json(
            evaluate(
                args.config,
                args.pred_dir,
                args.output_csv,
                valid_pred_dir=args.valid_pred_dir,
                cells=args.cells,
                perceptual=args.perceptual,
                efficiency_json=args.efficiency_json,
            )
        )
    elif args.command == "profile":
        from .profile import profile

        _print_json(profile(args.config, args.output_dir, device=args.device, dry_run=args.dry_run))
    elif args.command == "preprocess":
        from .preprocess import run_preprocessing

        _print_json(
            run_preprocessing(
                args.config,
                stages=args.stage or ["all"],
                output_dir=args.output_dir,
                overrides=args.set,
                dry_run=args.dry_run,
            )
        )
    else:
        parser.error(f"Unknown command: {args.command}")
