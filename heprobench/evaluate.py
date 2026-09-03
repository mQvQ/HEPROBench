from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .config import load_dataset_config
from .io_h5 import read_slide_h5
from .metrics import mae, mse, pearson, psnr, rmse, structural_similarity
from .schema import group_by_slide, load_channel_names, load_records
from .validate import validate_submission


BASE_METRICS = ["mae", "mse", "rmse", "pearson", "psnr", "ssim"]
PERCEPTUAL_METRICS = ["lpips", "dists"]
SUPPORTED_METRICS = set(BASE_METRICS + PERCEPTUAL_METRICS)


def _load_target(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with Image.open(path) as img:
        return np.asarray(img)


def _prediction_directory(path: str | Path, split: str) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / split).is_dir():
        return root / split
    return root


def _mean(rows: Iterable[dict[str, Any]], metric: str) -> float:
    values = np.asarray(
        [
            float(row[metric])
            for row in rows
            if row.get(metric) not in (None, "") and math.isfinite(float(row[metric]))
        ],
        dtype=np.float64,
    )
    return float(np.nanmean(values)) if len(values) else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _aggregate(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    count_name: str,
    metrics: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        result = dict(zip(group_keys, key))
        result[count_name] = len(group)
        result.update({metric: _mean(group, metric) for metric in metrics})
        output.append(result)
    return output


def _evaluation_settings(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        return {}
    from .experiment import load_experiment_config

    experiment, _ = load_experiment_config(path)
    evaluation = experiment.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    return evaluation


def _balanced_patch_indices(items: list[dict[str, Any]], sample_size: int, seed: int) -> list[int]:
    """Deterministically sample patches while giving each slide comparable representation."""

    if sample_size <= 0 or sample_size >= len(items):
        return list(range(len(items)))
    rng = np.random.default_rng(seed)
    by_slide: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_slide[str(item["slide_name"])].append(index)
    for indices in by_slide.values():
        rng.shuffle(indices)
    selected: list[int] = []
    slides = sorted(by_slide)
    while len(selected) < sample_size:
        progressed = False
        for slide_name in slides:
            indices = by_slide[slide_name]
            if indices:
                selected.append(indices.pop())
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:
            break
    return sorted(selected)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def evaluate(
    config_path: str | Path,
    pred_dir: str | Path,
    output_csv: str | Path | None = None,
    *,
    valid_pred_dir: str | Path | None = None,
    cells: bool | None = None,
    perceptual: bool | None = None,
    efficiency_json: str | Path | None = None,
) -> dict[str, Any]:
    evaluation_cfg = _evaluation_settings(config_path)
    image_cfg = evaluation_cfg.get("image", {})
    if image_cfg is None:
        image_cfg = {}
    if not isinstance(image_cfg, dict):
        raise ValueError("evaluation.image must be an object")
    configured_metrics = image_cfg.get("metrics", BASE_METRICS)
    if not isinstance(configured_metrics, list) or not all(isinstance(item, str) for item in configured_metrics):
        raise ValueError("evaluation.image.metrics must be a list of metric names")
    metrics = list(dict.fromkeys(item.lower() for item in configured_metrics))
    unknown_metrics = sorted(set(metrics).difference(SUPPORTED_METRICS))
    if unknown_metrics:
        raise ValueError(f"Unsupported image metrics: {unknown_metrics}")
    perceptual_cfg = image_cfg.get("perceptual", {})
    if perceptual_cfg is None:
        perceptual_cfg = {}
    if not isinstance(perceptual_cfg, dict):
        raise ValueError("evaluation.image.perceptual must be an object")
    run_perceptual = bool(perceptual_cfg.get("enabled", False)) if perceptual is None else bool(perceptual)
    if run_perceptual:
        for metric in PERCEPTUAL_METRICS:
            if metric not in metrics:
                metrics.append(metric)
    else:
        metrics = [metric for metric in metrics if metric not in PERCEPTUAL_METRICS]

    cfg, _ = load_dataset_config(config_path, split_override="test")
    dataset = cfg["dataset"]
    channel_names = load_channel_names(dataset["channel_names"])
    grouped = group_by_slide(load_records(dataset))
    test_pred_dir = _prediction_directory(pred_dir, "test")
    validate_submission(config_path, test_pred_dir, split_override="test")
    rows: list[dict[str, Any]] = []
    perceptual_items: list[dict[str, Any]] = []

    for slide_name, slide_records in grouped.items():
        payload = read_slide_h5(test_pred_dir / f"{slide_name}.h5")
        pred = payload["pred"]
        pred_by_coord = {
            (int(row), int(col)): pred[idx]
            for idx, (row, col) in enumerate(zip(payload["rows"], payload["cols"]))
        }
        for rec in slide_records:
            if rec.target_path is None:
                continue
            target = _load_target(rec.target_path)
            patch_pred = pred_by_coord[(rec.row, rec.col)]
            row_indices: list[int] = []
            for channel_idx, channel in enumerate(channel_names):
                prediction = patch_pred[..., channel_idx]
                truth = target[..., channel_idx]
                row_indices.append(len(rows))
                rows.append(
                    {
                        "slide_name": slide_name,
                        "row": rec.row,
                        "col": rec.col,
                        "channel": channel,
                        "mae": mae(prediction, truth),
                        "mse": mse(prediction, truth),
                        "rmse": rmse(prediction, truth),
                        "pearson": pearson(prediction, truth),
                        "psnr": psnr(prediction, truth),
                        "ssim": structural_similarity(prediction, truth),
                    }
                )
            if run_perceptual:
                perceptual_items.append(
                    {
                        "slide_name": slide_name,
                        "prediction": patch_pred.astype(np.uint8, copy=False),
                        "target": target.astype(np.uint8, copy=False),
                        "row_indices": row_indices,
                    }
                )

    perceptual_summary: dict[str, Any] = {"enabled": False}
    if run_perceptual:
        from .perceptual import perceptual_per_patch_channel

        if not perceptual_items:
            raise ValueError("Perceptual evaluation requires at least one prediction/target patch")
        sample_size = int(perceptual_cfg.get("sample_size", 3000))
        seed = int(perceptual_cfg.get("seed", 42))
        selected = _balanced_patch_indices(perceptual_items, sample_size, seed)
        selected_predictions = np.stack([perceptual_items[index]["prediction"] for index in selected])
        selected_targets = np.stack([perceptual_items[index]["target"] for index in selected])
        requested = set(metrics).intersection(PERCEPTUAL_METRICS)
        perceptual_values = perceptual_per_patch_channel(
            selected_predictions,
            selected_targets,
            metrics=requested,
            device=str(perceptual_cfg.get("device", "cpu")),
            patch_batch=int(perceptual_cfg.get("patch_batch", 16)),
            channel_chunk=int(perceptual_cfg.get("channel_chunk", 8)),
            lpips_net=str(perceptual_cfg.get("lpips_net", "alex")),
        )
        for result_index, item_index in enumerate(selected):
            for channel_index, row_index in enumerate(perceptual_items[item_index]["row_indices"]):
                for metric, values in perceptual_values.items():
                    rows[row_index][metric] = float(values[result_index, channel_index])
        perceptual_summary = {
            "enabled": True,
            "metrics": sorted(requested),
            "sampling": "slide-balanced deterministic sampling",
            "requested_patches": sample_size,
            "evaluated_patches": len(selected),
            "seed": seed,
            "device": str(perceptual_cfg.get("device", "cpu")),
            "lpips_net": str(perceptual_cfg.get("lpips_net", "alex")),
        }

    if output_csv is None:
        output_csv = test_pred_dir / "metrics.csv"
    output_csv = Path(output_csv)
    tile_fields = ["slide_name", "row", "col", "channel", *metrics]
    _write_csv(output_csv, rows, tile_fields)
    if output_csv.name != "metrics_tile_channel.csv":
        _write_csv(output_csv.parent / "metrics_tile_channel.csv", rows, tile_fields)

    slide_channel_rows = _aggregate(rows, ["slide_name", "channel"], "n_tiles", metrics)
    slide_rows = _aggregate(slide_channel_rows, ["slide_name"], "n_channels", metrics)
    _write_csv(
        output_csv.parent / "metrics_slide_channel.csv",
        slide_channel_rows,
        ["slide_name", "channel", "n_tiles", *metrics],
    )
    _write_csv(
        output_csv.parent / "metrics_slide.csv",
        slide_rows,
        ["slide_name", "n_channels", *metrics],
    )

    tile_weighted = {metric: _mean(rows, metric) for metric in metrics}
    slide_macro = {metric: _mean(slide_rows, metric) for metric in metrics}
    summary: dict[str, Any] = {
        # Keep the historical top-level fields backward compatible.
        **tile_weighted,
        "primary_aggregation": "slide_macro",
        "slide_macro": slide_macro,
        "tile_weighted": tile_weighted,
        "n_tile_channel_rows": len(rows),
        "n_slides": len(slide_rows),
        "image_metrics": metrics,
        "ssim_implementation": "skimage.metrics.structural_similarity, local-window, data_range=255",
        "perceptual": perceptual_summary,
        "metrics_csv": str(output_csv),
        "metrics_tile_channel_csv": str(output_csv.parent / "metrics_tile_channel.csv"),
        "metrics_slide_channel_csv": str(output_csv.parent / "metrics_slide_channel.csv"),
        "metrics_slide_csv": str(output_csv.parent / "metrics_slide.csv"),
    }

    cell_cfg = evaluation_cfg.get("cell", {})
    if cell_cfg is None:
        cell_cfg = {}
    if not isinstance(cell_cfg, dict):
        raise ValueError("evaluation.cell must be an object")
    run_cells = bool(cell_cfg.get("enabled", False)) if cells is None else bool(cells)
    if run_cells:
        from .cell_eval import evaluate_cells

        summary["cell"] = evaluate_cells(
            config_path,
            pred_dir,
            valid_pred_dir=valid_pred_dir,
            output_dir=output_csv.parent / "cell_metrics",
        )
    else:
        summary["cell"] = {"enabled": False}

    if efficiency_json is not None:
        efficiency_path = Path(efficiency_json).expanduser().resolve()
        with efficiency_path.open("r", encoding="utf-8") as handle:
            efficiency = json.load(handle)
        if not isinstance(efficiency, dict) or efficiency.get("schema") != "heprobench_efficiency_v1":
            raise ValueError(f"Unsupported efficiency report: {efficiency_path}")
        summary["computational_efficiency"] = {**efficiency, "source_json": str(efficiency_path)}
    else:
        summary["computational_efficiency"] = {
            "enabled": False,
            "note": "Run `heprobench profile` and pass --efficiency-json to include hardware-dependent results.",
        }

    summary = _json_safe(summary)
    with (output_csv.parent / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary
