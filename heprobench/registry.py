from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_REGISTRY_PATH = REPO_ROOT / "methods" / "registry.json"
ENCODER_REGISTRY_PATH = REPO_ROOT / "methods" / "pfm" / "specs.json"


@dataclass(frozen=True)
class TaskSpec:
    entrypoint: Path
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    paradigm: str
    aliases: tuple[str, ...]
    tasks: dict[str, TaskSpec]
    source: dict[str, Any]


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    display_name: str
    family: str
    source: str
    gated: bool
    checkpoint: dict[str, Any]
    benchmark_input_size: tuple[int, int]
    normalization: dict[str, Any]
    feature_layers: tuple[str, ...]


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Registry must contain a JSON object: {path}")
    return payload


def load_method_registry(path: str | Path | None = None) -> dict[str, MethodSpec]:
    source_path = Path(path).resolve() if path else METHOD_REGISTRY_PATH
    payload = _load_json_object(source_path)
    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, dict):
        raise ValueError(f"Method registry is missing the 'methods' object: {source_path}")

    methods: dict[str, MethodSpec] = {}
    for name, raw in raw_methods.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid method entry '{name}' in {source_path}")
        raw_tasks = raw.get("tasks", {})
        if not isinstance(raw_tasks, dict):
            raise ValueError(f"Invalid tasks for method '{name}'")
        tasks: dict[str, TaskSpec] = {}
        for task_name, task_raw in raw_tasks.items():
            if not isinstance(task_raw, dict) or not task_raw.get("entrypoint"):
                raise ValueError(f"Invalid task '{task_name}' for method '{name}'")
            entrypoint = Path(str(task_raw["entrypoint"]))
            if not entrypoint.is_absolute():
                entrypoint = REPO_ROOT / entrypoint
            raw_args = task_raw.get("args", [])
            if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
                raise ValueError(f"Invalid args for {name}.{task_name}")
            tasks[task_name] = TaskSpec(entrypoint=entrypoint.resolve(), args=tuple(raw_args))

        methods[name] = MethodSpec(
            name=name,
            display_name=str(raw.get("display_name", name)),
            paradigm=str(raw.get("paradigm", "unspecified")),
            aliases=tuple(str(item) for item in raw.get("aliases", [])),
            tasks=tasks,
            source=dict(raw.get("source", {})),
        )
    return methods


def resolve_method(name: str, path: str | Path | None = None) -> MethodSpec:
    methods = load_method_registry(path)
    key = name.strip().lower()
    if key in methods:
        return methods[key]
    for spec in methods.values():
        if key in {alias.lower() for alias in spec.aliases}:
            return spec
    available = ", ".join(sorted(methods))
    raise KeyError(f"Unknown method '{name}'. Available methods: {available}")


def load_encoder_registry(path: str | Path | None = None) -> dict[str, EncoderSpec]:
    source_path = Path(path).resolve() if path else ENCODER_REGISTRY_PATH
    payload = _load_json_object(source_path)
    raw_encoders = payload.get("encoders")
    if not isinstance(raw_encoders, dict):
        raise ValueError(f"Encoder registry is missing the 'encoders' object: {source_path}")

    encoders: dict[str, EncoderSpec] = {}
    for name, raw in raw_encoders.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid encoder entry '{name}' in {source_path}")
        encoders[name] = EncoderSpec(
            name=name,
            display_name=str(raw.get("display_name", name)),
            family=str(raw.get("family", "unspecified")),
            source=str(raw.get("source", "")),
            gated=bool(raw.get("gated", False)),
            checkpoint=dict(raw.get("checkpoint", {})),
            benchmark_input_size=tuple(int(item) for item in raw.get("benchmark_input_size", [256, 256])),
            normalization=dict(raw.get("normalization", {})),
            feature_layers=tuple(str(item) for item in raw.get("feature_layers", [])),
        )
    return encoders


def resolve_encoder(name: str, path: str | Path | None = None) -> EncoderSpec:
    encoders = load_encoder_registry(path)
    key = name.strip().lower()
    if key not in encoders:
        available = ", ".join(sorted(encoders))
        raise KeyError(f"Unknown foundation-model encoder '{name}'. Available encoders: {available}")
    return encoders[key]
