"""Small, dependency-free helpers used by method-specific runners.

The public JSON schema is shared, while every runner below deliberately calls
the method's own training or inference program.  This module only translates
configuration and records the resulting native command.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {
    "base_save_path",
    "augmentation_dir",
    "channel_names_file",
    "channel_stats_path",
    "checkpoint",
    "checkpoint_dir",
    "checkpoint_path",
    "checkpoints_dir",
    "csv_path",
    "dataset_config_path",
    "dataroot",
    "fm_features_path",
    "metadata",
    "out_root",
    "output_dir",
    "panel_dir",
    "prediction_dir",
    "pretrained_ckpt",
    "resume_path",
    "root_dir",
    "save_dir",
    "slide_dataframe_path",
    "src_folder",
    "test_dataframe_path",
    "test_csv",
    "tgt_folder",
    "tiling_dir",
    "train_dataframe_path",
    "train_csv",
    "val_dataframe_path",
    "val_csv",
    "vgg_path",
}


def parser(description: str, *, variant: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    if variant:
        result.add_argument("--variant", required=True)
    result.add_argument("task", choices=["train", "infer"])
    result.add_argument("--config", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    metadata = config.get("_heprobench", {})
    source = metadata.get("source_config") if isinstance(metadata, dict) else None
    source_path = Path(str(source)).expanduser().resolve() if source else config_path
    return config, source_path


def object_at(config: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key, {})
    if cursor is None:
        return {}
    if not isinstance(cursor, dict):
        raise ValueError(f"{'.'.join(keys)} must be a JSON object")
    return dict(cursor)


def native_task(config: Mapping[str, Any], task: str) -> dict[str, Any]:
    return object_at(config, "native", task)


def run_dir(config: Mapping[str, Any], task: str) -> Path:
    env_path = os.environ.get("HEPROBENCH_RUN_DIR")
    if env_path:
        result = Path(env_path)
    else:
        output = object_at(config, "output")
        result = Path(str(output.get("run_dir") or Path.cwd() / "outputs" / task))
    result = result.expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def resolve_path(value: Any, source_path: Path) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str((source_path.parent / expanded).resolve())


def resolve_known_paths(payload: Any, source_path: Path) -> Any:
    """Resolve path-valued fields without changing labels or model names."""

    if isinstance(payload, list):
        return [resolve_known_paths(item, source_path) for item in payload]
    if not isinstance(payload, dict):
        return payload
    resolved: dict[str, Any] = {}
    for key, value in payload.items():
        if key in PATH_KEYS:
            resolved[key] = resolve_path(value, source_path)
        else:
            resolved[key] = resolve_known_paths(value, source_path)
    return resolved


def first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def set_default(mapping: dict[str, Any], key: str, *values: Any) -> None:
    if key not in mapping or mapping[key] in (None, ""):
        value = first(*values)
        if value is not None:
            mapping[key] = value


def set_from(mapping: dict[str, Any], key: str, *values: Any) -> None:
    """Set ``key`` from the first shared value that is explicitly present.

    Method-specific ``native`` settings are defaults.  Shared top-level JSON
    fields are the public API and therefore must win when both are provided.
    """

    value = first(*values)
    if value is not None:
        mapping[key] = value


def split_csv(data: Mapping[str, Any], phase: str) -> Any:
    """Return the explicit CSV for a phase, falling back to ``csv_path``.

    A ``{split}`` placeholder is expanded here so native programs never need
    to know about the unified configuration convention.
    """

    aliases = {"val": "valid", "validation": "valid", "infer": "test"}
    normalized = aliases.get(phase, phase)
    value = first(data.get(f"{normalized}_csv"), data.get(f"{normalized}_dataframe_path"))
    if value is None:
        value = data.get("csv_path")
    if isinstance(value, str):
        return value.replace("{split}", normalized)
    return value


def deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def gpu_index(device: Any) -> str | None:
    if not isinstance(device, str):
        return None
    lowered = device.strip().lower()
    if lowered == "cpu":
        return "-1"
    if lowered.startswith("cuda:"):
        return lowered.split(":", 1)[1]
    if lowered == "cuda":
        return "0"
    return None


def cli_args(values: Mapping[str, Any], extra: Iterable[str] = ()) -> list[str]:
    """Turn a JSON object into conventional ``--key value`` arguments.

    ``true`` emits a flag and ``false``/``null`` omit it.  Native parsers that
    require an explicit false value can use the JSON string ``"false"``.
    """

    result: list[str] = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        option = key if key.startswith("-") else f"--{key}"
        if value is True:
            result.append(option)
        elif isinstance(value, list):
            result.append(option)
            result.extend(str(item) for item in value)
        else:
            result.extend([option, str(value)])
    result.extend(str(item) for item in extra)
    return result


def python_command(script: Path, args: Iterable[str], config: Mapping[str, Any]) -> list[str]:
    runtime = object_at(config, "runtime")
    executable = str(runtime.get("python") or sys.executable)
    return [executable, str(script.resolve()), *list(args)]


def training_command(
    script: Path,
    args: Iterable[str],
    config: Mapping[str, Any],
    task_config: Mapping[str, Any],
    *,
    module: str | None = None,
) -> list[str]:
    launcher = task_config.get("launcher", {})
    if launcher is None:
        launcher = {}
    if not isinstance(launcher, dict):
        raise ValueError("native.train.launcher must be an object")
    kind = str(launcher.get("type") or "python").lower()
    if kind == "python":
        return python_command(script, args, config)
    if kind != "torchrun":
        raise ValueError("native.train.launcher.type must be 'python' or 'torchrun'")
    runtime = object_at(config, "runtime")
    executable = str(runtime.get("python") or sys.executable)
    command = [executable, "-m", "torch.distributed.run"]
    if bool(launcher.get("standalone", True)):
        command.append("--standalone")
    command.extend(["--nproc_per_node", str(int(launcher.get("nproc_per_node", 1)))])
    if launcher.get("master_port"):
        command.extend(["--master_port", str(launcher["master_port"])])
    command.extend(["-m", module] if module else [str(script.resolve())])
    command.extend(list(args))
    return command


def write_json(payload: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def write_yaml_compatible_json(payload: Mapping[str, Any], destination: Path) -> Path:
    # JSON is valid YAML 1.2 and avoids importing the training stack in dry runs.
    return write_json(payload, destination)


def inference_payload(
    config: Mapping[str, Any],
    source_path: Path,
    *,
    method_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_config = native_task(config, "infer")
    native = task_config.get("config", {})
    if not isinstance(native, dict):
        raise ValueError("native.infer.config must be an object")
    payload = dict(native)
    data = object_at(config, "data")
    inference = object_at(config, "inference")
    model = object_at(config, "model")
    output = object_at(config, "output")
    runtime = object_at(config, "runtime")

    inference_split = str(inference.get("split") or data.get("split") or "test")
    set_from(payload, "csv_path", inference.get("csv_path"), split_csv(data, inference_split))
    set_from(payload, "root_dir", data.get("root_dir"))
    set_from(payload, "channel_names_file", data.get("channel_names_file"))
    set_from(payload, "split", inference_split)
    set_from(payload, "batch_size", inference.get("batch_size"))
    set_from(payload, "num_workers", inference.get("num_workers"))
    set_from(payload, "checkpoint_path", model.get("checkpoint_path"), model.get("checkpoint"))
    set_from(payload, "checkpoint_dir", model.get("checkpoint_dir"))
    set_from(payload, "out_root", output.get("prediction_dir"), output.get("pred_dir"))
    set_default(payload, "method_name", method_name)
    if "csv_columns" not in payload and isinstance(data.get("csv_columns"), dict):
        payload["csv_columns"] = data["csv_columns"]
    set_from(payload, "device", runtime.get("device"))
    return resolve_known_paths(payload, source_path), task_config


def emit_or_run(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    native_config: Path | None = None,
) -> int:
    run_path = Path(os.environ.get("HEPROBENCH_RUN_DIR") or cwd).resolve()
    plan: dict[str, Any] = {
        "command": shlex.join(command),
        "cwd": str(cwd.resolve()),
        "native_config": str(native_config) if native_config else "",
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    write_json(plan, run_path / "native_command.json")
    completed = subprocess.run(command, cwd=str(cwd), env=os.environ.copy(), check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return 0
