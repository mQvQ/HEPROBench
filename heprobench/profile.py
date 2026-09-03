from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_dataset_config
from .experiment import load_experiment_config
from .schema import load_channel_names


REPO_ROOT = Path(__file__).resolve().parents[1]


def _method_name(config: dict[str, Any]) -> str:
    method = config.get("method", {})
    if isinstance(method, str):
        return method.lower()
    if isinstance(method, dict):
        return str(method.get("name", "")).lower()
    return ""


def _commands(
    method: str,
    *,
    experiment: dict[str, Any],
    device: str,
    channels: int,
    region_size: int,
    patch_size: int,
    warmup: int,
    repeats: int,
    precision: str,
    params_csv: Path,
    latency_csv: Path,
) -> tuple[list[str], list[str], bool]:
    patches_per_region = max(1, (region_size // patch_size) ** 2)
    patch_iters = max(1, patches_per_region * repeats)
    common_params = ["--device", device, "--output-nc", str(channels), "--warmup", str(warmup), "--out-csv", str(params_csv)]
    common_time = [
        "--device", device, "--channels", str(channels), "--warmup", str(warmup),
        "--precision", precision, "--out-csv", str(latency_csv),
    ]

    if method == "rosie":
        base = REPO_ROOT / "methods" / "rosie" / "implementation"
        params = [
            sys.executable, str(base / "measure_params_flops.py"), "--img-size", "224",
            "--torchvision-weights", "none", *common_params,
        ]
        latency = [
            sys.executable, str(base / "measure_infer_region_time.py"), "--region-size", str(region_size),
            "--patch-size", str(patch_size), "--iters", str(repeats),
            "--torchvision-weights", "none", *common_time,
        ]
        return params, latency, True
    if method == "hex":
        base = REPO_ROOT / "methods" / "hex" / "implementation"
        params = [sys.executable, str(base / "measure_params_flops.py"), "--img-size", "384", *common_params]
        latency = [
            sys.executable, str(base / "hex" / "measure_infer_region_time.py"),
            "--region-size", str(region_size), "--patch-size", str(patch_size), "--iters", str(repeats),
            *common_time,
        ]
        return params, latency, True
    if method == "cut":
        base = REPO_ROOT / "methods" / "cut" / "implementation"
        params = [sys.executable, str(base / "measure_params_flops.py"), "--img-size", str(patch_size), *common_params]
        latency = [
            sys.executable, str(base / "measure_infer_time.py"), "--img-size", str(patch_size),
            "--batch-size", "1", "--iters", str(patch_iters), *common_time,
        ]
        return params, latency, False
    if method == "histoplexer":
        base = REPO_ROOT / "methods" / "histoplexer" / "implementation"
        params = [sys.executable, str(base / "measure_params_flops.py"), "--img-size", str(patch_size), *common_params]
        latency = [
            sys.executable, str(base / "measure_infer_time.py"), "--img-size", str(patch_size),
            "--patch-size", str(patch_size), "--batch-size", "1", "--iters", str(patch_iters), *common_time,
        ]
        return params, latency, False
    if method in {"pix2pix", "cyclegan"}:
        model = "pix2pix" if method == "pix2pix" else "cyclegan"
        base = REPO_ROOT / "methods" / "pytorch_cyclegan_and_pix2pix" / "implementation"
        params = [
            sys.executable, str(base / "measure_params_flops.py"), "--model", model,
            "--img-size", str(patch_size), *common_params,
        ]
        latency = [
            sys.executable, str(base / "measure_infer_time.py"), "--model", model,
            "--img-size", str(patch_size), "--batch-size", "1", "--iters", str(patch_iters), *common_time,
        ]
        return params, latency, False
    if method in {"miphei_vit", "gigatime_reg", "dpt_fm"}:
        base = REPO_ROOT / "methods" / "miphei_vit" / "implementation" / "hepro"
        model_cfg = experiment.get("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        encoder_cfg = model_cfg.get("encoder", {})
        if not isinstance(encoder_cfg, dict):
            encoder_cfg = {}
        model_name = "dpt" if method == "dpt_fm" else str(model_cfg.get("model_name") or ("gigatime" if method == "gigatime_reg" else "myvitmatte"))
        encoder_name = str(encoder_cfg.get("name") or encoder_cfg.get("encoder_name") or "h0-mini")
        shared = [
            "--model-name", model_name, "--encoder-name", encoder_name, "--device", device,
            "--img-size", str(patch_size), "--output-nc", str(channels), "--precision", precision,
            "--warmup", str(warmup),
        ]
        if bool(encoder_cfg.get("frozen", method == "dpt_fm")):
            shared.append("--encoder-frozen")
        if bool(model_cfg.get("use_lora", False)):
            shared.append("--use-lora")
        script = base / "measure_efficiency.py"
        params = [sys.executable, str(script), "--mode", "params", *shared, "--out-csv", str(params_csv)]
        latency = [
            sys.executable, str(script), "--mode", "latency", *shared,
            "--batch-size", "1", "--iters", str(patch_iters), "--out-csv", str(latency_csv),
        ]
        return params, latency, False
    if method == "gigatime_original":
        script = REPO_ROOT / "methods" / "gigatime" / "implementation" / "original" / "measure_efficiency.py"
        shared = [
            "--device", device, "--img-size", "556", "--output-nc", str(channels),
            "--precision", precision, "--warmup", str(warmup),
        ]
        params = [sys.executable, str(script), "--mode", "params", *shared, "--out-csv", str(params_csv)]
        latency = [
            sys.executable, str(script), "--mode", "latency", *shared,
            "--batch-size", "1", "--iters", str(patch_iters), "--out-csv", str(latency_csv),
        ]
        return params, latency, False
    raise NotImplementedError(f"Unified native profiling is not registered for method={method!r}")


def _read_single_row(path: Path, channels: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Profiler produced no rows: {path}")
    for row in rows:
        value = row.get("nc_out") or row.get("output_nc") or row.get("channels")
        if value not in (None, "") and int(value) == channels:
            return row
    return rows[0]


def _number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def profile(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    experiment, _ = load_experiment_config(config_path)
    method = _method_name(experiment)
    dataset_cfg, _ = load_dataset_config(config_path, split_override="test")
    channel_names = load_channel_names(dataset_cfg["dataset"]["channel_names"])
    evaluation = experiment.get("evaluation", {})
    efficiency = evaluation.get("efficiency", {}) if isinstance(evaluation, dict) else {}
    if not isinstance(efficiency, dict):
        raise ValueError("evaluation.efficiency must be an object")
    runtime = experiment.get("runtime", {})
    chosen_device = str(device or (runtime.get("device") if isinstance(runtime, dict) else None) or "cpu")
    if chosen_device == "cpu":
        precision = "fp32"
    else:
        precision = str(efficiency.get("precision", "fp16"))
    region_size = int(efficiency.get("region_size", 2048))
    patch_size = int(efficiency.get("patch_size", 256))
    warmup = int(efficiency.get("warmup", 1))
    repeats = int(efficiency.get("repeats", 2))

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    params_csv = output_dir / "params_flops.csv"
    latency_csv = output_dir / "latency.csv"
    params_command, latency_command, direct_region_timing = _commands(
        method,
        experiment=experiment,
        device=chosen_device,
        channels=len(channel_names),
        region_size=region_size,
        patch_size=patch_size,
        warmup=warmup,
        repeats=repeats,
        precision=precision,
        params_csv=params_csv,
        latency_csv=latency_csv,
    )
    result: dict[str, Any] = {
        "schema": "heprobench_efficiency_v1",
        "enabled": True,
        "method": method,
        "protocol": str(efficiency.get("protocol", "paper")),
        "device": chosen_device,
        "precision": precision,
        "batch_size": 1,
        "region_size": region_size,
        "patch_size": patch_size,
        "output_channels": len(channel_names),
        "warmup": warmup,
        "repeats": repeats,
        "commands": [shlex.join(params_command), shlex.join(latency_command)],
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return result

    for command in (params_command, latency_command):
        subprocess.run(command, cwd=str(Path(command[1]).parent), check=True)
    params_row = _read_single_row(params_csv, len(channel_names))
    latency_row = _read_single_row(latency_csv, len(channel_names))
    patches_per_region = max(1, (region_size // patch_size) ** 2)
    if direct_region_timing:
        mean_ms = _number(latency_row, "mean_ms_region")
        p50_ms = _number(latency_row, "p50_ms_region")
        p90_ms = _number(latency_row, "p90_ms_region")
        p99_ms = _number(latency_row, "p99_ms_region")
    else:
        mean_patch = _number(latency_row, "mean_ms_patch")
        mean_ms = mean_patch * patches_per_region if mean_patch is not None else None
        p50_patch = _number(latency_row, "p50_ms_batch")
        p90_patch = _number(latency_row, "p90_ms_batch")
        p99_patch = _number(latency_row, "p99_ms_batch")
        p50_ms = p50_patch * patches_per_region if p50_patch is not None else None
        p90_ms = p90_patch * patches_per_region if p90_patch is not None else None
        p99_ms = p99_patch * patches_per_region if p99_patch is not None else None
    flops = _number(params_row, "flops")
    result.update(
        {
            "params_total": int(_number(params_row, "params_total") or 0),
            "params_trainable": int(_number(params_row, "params_trainable") or 0),
            "flops_per_patch": int(flops) if flops is not None else None,
            "gflops_per_patch": flops / 1e9 if flops is not None else None,
            "latency_mean_ms_per_region": mean_ms,
            "latency_p50_ms_per_region": p50_ms,
            "latency_p90_ms_per_region": p90_ms,
            "latency_p99_ms_per_region": p99_ms,
            "params_flops_csv": str(params_csv),
            "latency_csv": str(latency_csv),
        }
    )
    output_json = output_dir / "efficiency.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    result["efficiency_json"] = str(output_json)
    return result
