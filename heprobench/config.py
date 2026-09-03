from __future__ import annotations

from pathlib import Path
from typing import Any

from .experiment import load_experiment_config


def load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and return the data with the absolute source path."""
    import yaml

    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {source_path}")
    return data, source_path


def resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_dataset_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    requested_path = Path(path).expanduser().resolve()
    if requested_path.suffix.lower() == ".json":
        experiment, source_path = load_experiment_config(requested_path)
        data = experiment.get("data", {})
        inference = experiment.get("inference", {})
        evaluation = experiment.get("evaluation", {})
        if not isinstance(data, dict) or not isinstance(inference, dict) or not isinstance(evaluation, dict):
            raise ValueError("Unified config data/inference/evaluation sections must be objects")
        split = str(evaluation.get("split") or inference.get("split") or data.get("split") or "test")
        metadata = evaluation.get("csv_path") or inference.get("csv_path") or data.get("csv_path")
        if not metadata:
            metadata = data.get(f"{split}_dataframe_path")
        if not metadata:
            raise ValueError("Unified evaluation requires data.csv_path or evaluation.csv_path")
        if isinstance(metadata, str):
            metadata = metadata.format(split=split)
        columns = data.get("csv_columns", {})
        if not isinstance(columns, dict):
            raise ValueError("data.csv_columns must be an object")
        cfg = {
            "dataset": {
                "name": data.get("cohort", "heprobench"),
                "root": data.get("root_dir", "."),
                "metadata_csv": metadata,
                "channel_names": data.get("channel_names_file", "channel_names.json"),
                "image_column": columns.get("image_path", "image_path"),
                "target_column": columns.get("target_path", "target_path"),
                "slide_column": columns.get("slide_name", "slide_name"),
                "row_column": columns.get("row", "row"),
                "col_column": columns.get("col", "col"),
                "split_column": columns.get("split", "split"),
                "split": split,
                "patch_size": data.get("patch_size", 256),
            }
        }
    else:
        cfg, source_path = load_yaml(requested_path)
    dataset = cfg.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset config must contain a 'dataset' mapping")
    config_dir = source_path.parent
    root = resolve_path(dataset.get("root", "."), config_dir)
    dataset["root"] = str(root)
    dataset["metadata_csv"] = str(resolve_path(dataset.get("metadata_csv", "metadata.csv"), root))
    dataset["channel_names"] = str(resolve_path(dataset.get("channel_names", "channel_names.json"), root))
    dataset.setdefault("image_column", "image_path")
    dataset.setdefault("target_column", "target_path")
    dataset.setdefault("slide_column", "slide_name")
    dataset.setdefault("row_column", "row")
    dataset.setdefault("col_column", "col")
    dataset.setdefault("split_column", "split")
    dataset.setdefault("patch_size", 256)
    return cfg, source_path


def load_method_config(method: str | Path, dataset_config_path: Path) -> tuple[dict[str, Any], Path]:
    method_path = Path(method).expanduser()
    if method_path.suffix not in {".yaml", ".yml"}:
        candidates = [
            Path.cwd() / "configs" / "methods" / f"{method}.yaml",
            dataset_config_path.parent / "methods" / f"{method}.yaml",
            dataset_config_path.parent.parent / "configs" / "methods" / f"{method}.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                method_path = candidate
                break
        else:
            raise FileNotFoundError(f"Cannot find method config for '{method}'")
    if not method_path.is_absolute():
        method_path = (Path.cwd() / method_path).resolve()

    cfg, source_path = load_yaml(method_path)
    method_cfg = cfg.setdefault("method", {})
    if not isinstance(method_cfg, dict):
        raise ValueError("method config must contain a 'method' mapping")
    checkpoint = method_cfg.get("checkpoint")
    if checkpoint:
        method_cfg["checkpoint"] = str(resolve_path(checkpoint, source_path.parent))
    return cfg, source_path
