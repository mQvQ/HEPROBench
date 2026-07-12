from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate
from .infer import run_inference
from .train import run_training
from .validate import validate_data, validate_submission


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HEPROBench review demo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data", help="Validate dataset metadata and target arrays")
    p.add_argument("--config", required=True)
    p.add_argument("--skip-arrays", action="store_true")

    p = sub.add_parser("infer", help="Run a method config and write HDF5 submission files")
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True, help="Method config path or name under configs/methods")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("train", help="Train a tiny method checkpoint on the demo dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True, help="Method config path or name under configs/methods")
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("validate-submission", help="Validate HDF5 submission files")
    p.add_argument("--config", required=True)
    p.add_argument("--pred-dir", required=True)

    p = sub.add_parser("evaluate", help="Evaluate a HDF5 submission against demo targets")
    p.add_argument("--config", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--output-csv", default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-data":
        _print_json(validate_data(args.config, check_arrays=not args.skip_arrays))
    elif args.command == "infer":
        _print_json(run_inference(args.config, args.method, Path(args.output), device=args.device))
    elif args.command == "train":
        _print_json(run_training(args.config, args.method, args.output_checkpoint, epochs=args.epochs, lr=args.lr, device=args.device))
    elif args.command == "validate-submission":
        _print_json(validate_submission(args.config, args.pred_dir))
    elif args.command == "evaluate":
        _print_json(evaluate(args.config, args.pred_dir, args.output_csv))
    else:
        parser.error(f"Unknown command: {args.command}")

