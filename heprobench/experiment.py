from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "heprobench_experiment_v1"


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_bases(source: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if source in stack:
        chain = " -> ".join(str(item) for item in (*stack, source))
        raise ValueError(f"Circular experiment config inheritance: {chain}")
    if source.suffix.lower() != ".json":
        raise ValueError(f"Unified method runs require a JSON config: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must contain a JSON object: {source}")
    raw_bases = payload.pop("extends", [])
    if isinstance(raw_bases, str):
        raw_bases = [raw_bases]
    if not isinstance(raw_bases, list) or not all(isinstance(item, str) for item in raw_bases):
        raise ValueError(f"'extends' must be a path or list of paths: {source}")
    merged: dict[str, Any] = {}
    for item in raw_bases:
        base_path = Path(item).expanduser()
        if not base_path.is_absolute():
            base_path = source.parent / base_path
        merged = _merge_dicts(merged, _load_with_bases(base_path.resolve(), (*stack, source)))
    return _merge_dicts(merged, payload)


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    return _load_with_bases(source), source


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    keys = dotted_key.split(".")
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set '{dotted_key}': '{key}' is not an object")
        cursor = existing
    cursor[keys[-1]] = value


def parse_overrides(items: list[str] | None) -> list[tuple[str, Any]]:
    parsed: list[tuple[str, Any]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        key, raw_value = item.split("=", 1)
        parsed.append((key.strip(), _json_value(raw_value.strip())))
    return parsed


def _method_name(config: dict[str, Any]) -> str | None:
    method = config.get("method")
    if isinstance(method, str):
        return method
    if isinstance(method, dict) and method.get("name"):
        return str(method["name"])
    return None


def resolve_experiment_config(
    config: dict[str, Any],
    *,
    source_path: Path,
    method: str,
    task: str,
    encoder: str | None = None,
    split: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    output_dir: str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    configured_method = _method_name(resolved)
    if configured_method and configured_method.lower() != method.lower():
        raise ValueError(
            f"CLI method '{method}' does not match config method '{configured_method}' in {source_path}"
        )
    method_block = resolved.setdefault("method", {})
    if isinstance(method_block, str):
        method_block = {"name": method_block}
        resolved["method"] = method_block
    if not isinstance(method_block, dict):
        raise ValueError("'method' must be a string or JSON object")
    method_block["name"] = method

    resolved.setdefault("schema_version", SCHEMA_VERSION)
    if resolved["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={resolved['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )

    if encoder is not None:
        model = resolved.setdefault("model", {})
        if not isinstance(model, dict):
            raise ValueError("'model' must be a JSON object")
        encoder_block = model.setdefault("encoder", {})
        if not isinstance(encoder_block, dict):
            raise ValueError("'model.encoder' must be a JSON object")
        encoder_block["name"] = encoder
    if split is not None:
        section = resolved.setdefault("inference" if task == "infer" else "data", {})
        if not isinstance(section, dict):
            raise ValueError("Task configuration section must be a JSON object")
        section["split"] = split
    if device is not None:
        runtime = resolved.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("'runtime' must be a JSON object")
        runtime["device"] = device
    if batch_size is not None:
        section = resolved.setdefault("train" if task == "train" else "inference", {})
        if not isinstance(section, dict):
            raise ValueError("Task configuration section must be a JSON object")
        section["batch_size"] = int(batch_size)
    if output_dir is not None:
        output = resolved.setdefault("output", {})
        if not isinstance(output, dict):
            raise ValueError("'output' must be a JSON object")
        output["run_dir"] = output_dir
    for key, value in parse_overrides(overrides):
        set_dotted(resolved, key, value)

    metadata = resolved.setdefault("_heprobench", {})
    if not isinstance(metadata, dict):
        raise ValueError("'_heprobench' is reserved for framework metadata")
    metadata.update(
        {
            "source_config": str(source_path),
            "task": task,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    metadata["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return resolved


def get_run_dir(config: dict[str, Any], *, method: str, task: str, source_path: Path) -> Path:
    output = config.get("output", {})
    if output is not None and not isinstance(output, dict):
        raise ValueError("'output' must be a JSON object")
    configured = (output or {}).get("run_dir")
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = source_path.parent / path
        return path.resolve()
    run_name = str((output or {}).get("run_name") or source_path.stem)
    return (Path.cwd() / "outputs" / method / run_name / task).resolve()


def write_resolved_config(config: dict[str, Any], run_dir: Path, task: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / f"resolved_config.{task}.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
