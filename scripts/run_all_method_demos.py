#!/usr/bin/env python3
"""Run and verify every native method demo through the unified CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from heprobench.experiment import get_run_dir, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DemoMethod:
    name: str
    config: str
    prediction_name: str
    checkpoints: tuple[str, ...]


DEMO_METHODS = (
    DemoMethod("rosie", "configs/demo/rosie.json", "ROSIE", ("checkpoints/rosie-demo/latest_model.pth",)),
    DemoMethod("cut", "configs/demo/cut.json", "CUT", ("checkpoints/cut-demo/latest_net_G.pth",)),
    DemoMethod("pix2pix", "configs/demo/pix2pix.json", "pix2pix", ("checkpoints/pix2pix-demo/latest_net_G.pth",)),
    DemoMethod("cyclegan", "configs/demo/cyclegan.json", "cyclegan", ("checkpoints/cyclegan-demo/latest_net_G_A.pth",)),
    DemoMethod(
        "histoplexer",
        "configs/demo/histoplexer.json",
        "HistoPlexer",
        ("synthetic-demo_ours_imc01_channels-all_seed-96/checkpoint-step_1.pt",),
    ),
    DemoMethod(
        "gigatime_original",
        "configs/demo/gigatime_original.json",
        "GigaTIME-Original",
        ("models/gigatime-original-demo/model.pth",),
    ),
    DemoMethod(
        "gigatime_reg",
        "configs/demo/gigatime_reg.json",
        "GigaTIME-Reg",
        ("checkpoint/config.yaml", "checkpoint/model.weights.ckpt"),
    ),
    DemoMethod(
        "miphei_vit",
        "configs/demo/miphei_vit.json",
        "MIPHEI-ViT",
        ("checkpoint/config.yaml", "checkpoint/model.weights.ckpt"),
    ),
    DemoMethod(
        "dpt_fm",
        "configs/demo/dpt_fm_h0-mini.json",
        "DPT-h0-mini",
        ("checkpoint/config.yaml", "checkpoint/model.weights.ckpt"),
    ),
    DemoMethod("hex", "configs/demo/hex.json", "HEX", ("checkpoints/checkpoint_step_1.pth",)),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _command(step: str, command: list[str]) -> dict[str, Any]:
    print(f"\n[{step}] {' '.join(command)}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {
        "step": step,
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": "passed" if completed.returncode == 0 else "failed",
    }


def _artifact_step(step: str, paths: list[Path]) -> dict[str, Any]:
    missing = [str(path) for path in paths if not path.is_file()]
    return {
        "step": step,
        "status": "passed" if not missing else "failed",
        "paths": [str(path) for path in paths],
        "missing": missing,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Override every demo device, e.g. cuda:0")
    parser.add_argument(
        "--method",
        action="append",
        choices=[item.name for item in DEMO_METHODS],
        help="Run one method; repeat to select several. The default runs all methods.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoints")
    parser.add_argument("--skip-infer", action="store_true", help="Reuse existing valid/test predictions")
    parser.add_argument("--perceptual", action="store_true", help="Also calculate LPIPS and DISTS")
    parser.add_argument("--profile", action="store_true", help="Run native efficiency profiling and attach it to evaluation")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs" / "demo" / "verification_summary.json",
        help="Machine-readable verification report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected = set(args.method or [item.name for item in DEMO_METHODS])
    methods = [item for item in DEMO_METHODS if item.name in selected]
    summary_path = args.summary.expanduser().resolve()
    summary: dict[str, Any] = {
        "schema": "heprobench_demo_verification_v1",
        "started_at": _utc_now(),
        "repository": str(ROOT),
        "python": sys.executable,
        "device_override": args.device,
        "perceptual": bool(args.perceptual),
        "profile": bool(args.profile),
        "status": "running",
        "methods": [],
    }
    _write_summary(summary_path, summary)
    all_passed = True

    for method in methods:
        config_path = ROOT / method.config
        config, source = load_experiment_config(config_path)
        run_dir = get_run_dir(config, method=method.name, task="train", source_path=source)
        prediction_root = run_dir / "predictions" / method.prediction_name
        method_result: dict[str, Any] = {
            "method": method.name,
            "config": str(config_path),
            "run_dir": str(run_dir),
            "prediction_root": str(prediction_root),
            "status": "running",
            "steps": [],
        }
        summary["methods"].append(method_result)
        print(f"\n{'=' * 72}\n{method.name}\n{'=' * 72}", flush=True)

        commands: list[tuple[str, list[str]]] = []
        common = [sys.executable, "-m", "heprobench"]
        device_args = ["--device", args.device] if args.device else []
        if not args.skip_train:
            commands.append(("train", [*common, "train", "--config", method.config, *device_args]))

        method_passed = True
        for step, command in commands:
            result = _command(step, command)
            method_result["steps"].append(result)
            _write_summary(summary_path, summary)
            if result["status"] != "passed":
                method_passed = False
                break

        if method_passed:
            checkpoint_result = _artifact_step(
                "checkpoint_contract",
                [run_dir / relative for relative in method.checkpoints],
            )
            method_result["steps"].append(checkpoint_result)
            method_passed = checkpoint_result["status"] == "passed"
            _write_summary(summary_path, summary)

        if method_passed and not args.skip_infer:
            for split in ("valid", "test"):
                result = _command(
                    f"infer_{split}",
                    [
                        *common,
                        "infer",
                        "--config",
                        method.config,
                        "--split",
                        split,
                        *device_args,
                        "--set",
                        "native.infer.config.overwrite=true",
                    ],
                )
                method_result["steps"].append(result)
                _write_summary(summary_path, summary)
                if result["status"] != "passed":
                    method_passed = False
                    break

        if method_passed:
            for split in ("valid", "test"):
                result = _command(
                    f"validate_{split}",
                    [
                        *common,
                        "validate-submission",
                        "--config",
                        method.config,
                        "--pred-dir",
                        str(prediction_root / split),
                        "--split",
                        split,
                    ],
                )
                method_result["steps"].append(result)
                _write_summary(summary_path, summary)
                if result["status"] != "passed":
                    method_passed = False
                    break

        efficiency_path: Path | None = None
        if method_passed and args.profile:
            profile_dir = run_dir / "profile"
            result = _command(
                "profile",
                [*common, "profile", "--config", method.config, "--output-dir", str(profile_dir), *device_args],
            )
            method_result["steps"].append(result)
            method_passed = result["status"] == "passed"
            efficiency_path = profile_dir / "efficiency.json"
            _write_summary(summary_path, summary)

        if method_passed:
            evaluation_command = [
                *common,
                "evaluate",
                "--config",
                method.config,
                "--pred-dir",
                str(prediction_root),
                "--perceptual" if args.perceptual else "--no-perceptual",
            ]
            if efficiency_path is not None:
                evaluation_command.extend(["--efficiency-json", str(efficiency_path)])
            result = _command("evaluate", evaluation_command)
            method_result["steps"].append(result)
            method_passed = result["status"] == "passed"
            _write_summary(summary_path, summary)

        if method_passed:
            artifact_result = _artifact_step(
                "evaluation_contract",
                [
                    prediction_root / "test" / "summary.json",
                    prediction_root / "test" / "metrics_tile_channel.csv",
                    prediction_root / "test" / "metrics_slide.csv",
                    prediction_root / "test" / "cell_metrics" / "cell_pcc_summary.json",
                    prediction_root / "test" / "cell_metrics" / "cell_classification_summary.json",
                ],
            )
            method_result["steps"].append(artifact_result)
            method_passed = artifact_result["status"] == "passed"

        method_result["status"] = "passed" if method_passed else "failed"
        all_passed = all_passed and method_passed
        _write_summary(summary_path, summary)

    summary["finished_at"] = _utc_now()
    summary["status"] = "passed" if all_passed else "failed"
    summary["passed_methods"] = sum(item["status"] == "passed" for item in summary["methods"])
    summary["total_methods"] = len(summary["methods"])
    _write_summary(summary_path, summary)
    print(f"\nVerification {summary['status']}: {summary['passed_methods']}/{summary['total_methods']} methods")
    print(f"Summary: {summary_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
