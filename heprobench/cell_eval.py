from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import load_dataset_config
from .io_h5 import read_slide_h5
from .metrics import pearson
from .schema import PatchRecord, group_by_slide, load_channel_names, load_records


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _prediction_directory(path: str | Path, split: str) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / split).is_dir():
        return root / split
    if root.name == split and root.is_dir():
        return root
    return root


def _accumulate_patch(
    accumulator: dict[tuple[str, int], tuple[np.ndarray, int]],
    *,
    slide_name: str,
    mask: np.ndarray,
    values: np.ndarray,
) -> None:
    if mask.shape != values.shape[:2]:
        raise ValueError(f"Mask/value shape mismatch for {slide_name}: {mask.shape} vs {values.shape[:2]}")
    ids = mask.astype(np.int64, copy=False).reshape(-1)
    pixels = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
    unique, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((len(unique), pixels.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, pixels)
    counts = np.bincount(inverse, minlength=len(unique))
    for index, cell_id in enumerate(unique.tolist()):
        if int(cell_id) == 0:
            continue
        key = (slide_name, int(cell_id))
        previous = accumulator.get(key)
        if previous is None:
            accumulator[key] = (sums[index], int(counts[index]))
        else:
            accumulator[key] = (previous[0] + sums[index], previous[1] + int(counts[index]))


def aggregate_prediction_cells(
    pred_dir: str | Path,
    records: Iterable[PatchRecord],
    channel_names: list[str],
) -> dict[tuple[str, int], np.ndarray]:
    pred_dir = Path(pred_dir)
    accumulator: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    for slide_name, slide_records in group_by_slide(records).items():
        payload = read_slide_h5(pred_dir / f"{slide_name}.h5")
        if payload["channel_names"] != channel_names:
            raise ValueError(f"Channel mismatch in {pred_dir / f'{slide_name}.h5'}")
        pred_by_coord = {
            (int(row), int(col)): payload["pred"][index]
            for index, (row, col) in enumerate(zip(payload["rows"], payload["cols"]))
        }
        for record in slide_records:
            if record.mask_path is None or not record.mask_path.exists():
                raise FileNotFoundError(f"Cell mask is required for cell evaluation: {record.mask_path}")
            _accumulate_patch(
                accumulator,
                slide_name=slide_name,
                mask=np.load(record.mask_path),
                values=pred_by_coord[(record.row, record.col)],
            )
    return {key: value_sum / count for key, (value_sum, count) in accumulator.items() if count > 0}


def _load_annotations(path: Path, channel_names: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    annotations: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"slide_name", "global_cell_id", "split", *channel_names}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Cell annotation CSV missing columns: {', '.join(missing)}")
        for row in reader:
            key = (str(row["slide_name"]), int(row["global_cell_id"]))
            if key in annotations:
                raise ValueError(f"Duplicated cell annotation: {key}")
            annotations[key] = dict(row)
    return annotations


def _cell_rows(
    values: dict[tuple[str, int], np.ndarray],
    annotations: dict[tuple[str, int], dict[str, Any]],
    channel_names: list[str],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (slide_name, cell_id), prediction in sorted(values.items()):
        annotation = annotations.get((slide_name, cell_id))
        if annotation is None or str(annotation.get("split")) != split:
            continue
        row: dict[str, Any] = {"slide_name": slide_name, "global_cell_id": cell_id, "split": split}
        for index, marker in enumerate(channel_names):
            row[f"pred_{marker}"] = float(prediction[index])
            row[f"gt_{marker}"] = float(annotation[marker])
            label = annotation.get(f"{marker}_pos")
            if label not in (None, ""):
                row[f"label_{marker}"] = int(label)
        rows.append(row)
    return rows


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_cells(
    config_path: str | Path,
    pred_dir: str | Path,
    *,
    valid_pred_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    test_cfg, source = load_dataset_config(config_path, split_override="test")
    valid_cfg, _ = load_dataset_config(config_path, split_override="valid")
    test_dataset = test_cfg["dataset"]
    valid_dataset = valid_cfg["dataset"]
    channel_names = load_channel_names(test_dataset["channel_names"])

    experiment_path = Path(config_path).expanduser().resolve()
    with experiment_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle) if experiment_path.suffix.lower() == ".json" else {}
    # Evaluation settings may be inherited, so read the fully resolved config.
    if experiment_path.suffix.lower() == ".json":
        from .experiment import load_experiment_config

        raw, source = load_experiment_config(experiment_path)
    evaluation = raw.get("evaluation", {}) if isinstance(raw, dict) else {}
    cell_cfg = evaluation.get("cell", {}) if isinstance(evaluation, dict) else {}
    if not isinstance(cell_cfg, dict):
        raise ValueError("evaluation.cell must be an object")

    annotations_value = cell_cfg.get("annotations_csv")
    if not annotations_value:
        raise ValueError("evaluation.cell.annotations_csv is required when cell evaluation is enabled")
    annotations_path = Path(str(annotations_value)).expanduser()
    if not annotations_path.is_absolute():
        annotations_path = source.parent / annotations_path
    annotations_path = annotations_path.resolve()

    test_records = load_records(test_dataset)
    valid_records = load_records(valid_dataset)
    test_pred = _prediction_directory(pred_dir, "test")
    if valid_pred_dir is None:
        candidate_root = Path(pred_dir).expanduser().resolve()
        candidate = candidate_root / "valid" if candidate_root.name != "test" else candidate_root.parent / "valid"
        valid_pred_dir = candidate
    valid_pred = _prediction_directory(valid_pred_dir, "valid")

    annotations = _load_annotations(annotations_path, channel_names)
    test_values = aggregate_prediction_cells(test_pred, test_records, channel_names)
    valid_values = aggregate_prediction_cells(valid_pred, valid_records, channel_names)
    test_rows = _cell_rows(test_values, annotations, channel_names, "test")
    valid_rows = _cell_rows(valid_values, annotations, channel_names, "valid")
    if not test_rows or not valid_rows:
        raise ValueError("Cell evaluation requires non-empty valid and test cell predictions")

    output_dir = Path(output_dir)
    value_fields = ["slide_name", "global_cell_id", "split"]
    for marker in channel_names:
        value_fields.extend([f"pred_{marker}", f"gt_{marker}", f"label_{marker}"])
    _write_csv(output_dir / "cell_values_valid.csv", valid_rows, value_fields)
    _write_csv(output_dir / "cell_values_test.csv", test_rows, value_fields)

    pcc_rows: list[dict[str, Any]] = []
    for slide_name in sorted({str(row["slide_name"]) for row in test_rows}):
        slide_rows = [row for row in test_rows if row["slide_name"] == slide_name]
        for marker in channel_names:
            pred = np.asarray([row[f"pred_{marker}"] for row in slide_rows])
            target = np.asarray([row[f"gt_{marker}"] for row in slide_rows])
            pcc_rows.append(
                {
                    "slide_name": slide_name,
                    "marker": marker,
                    "n_cells": len(slide_rows),
                    "pcc": pearson(pred, target),
                }
            )
    _write_csv(
        output_dir / "cell_pcc_per_slide_marker.csv",
        pcc_rows,
        ["slide_name", "marker", "n_cells", "pcc"],
    )
    pcc_per_marker: dict[str, dict[str, Any]] = {}
    for marker in channel_names:
        values = np.asarray([row["pcc"] for row in pcc_rows if row["marker"] == marker], dtype=np.float64)
        finite = values[np.isfinite(values)]
        pcc_per_marker[marker] = {
            "n_slides": int(len(values)),
            "n_finite": int(len(finite)),
            "mean": _finite_or_none(float(np.mean(finite))) if len(finite) else None,
            "median": _finite_or_none(float(np.median(finite))) if len(finite) else None,
            "std": _finite_or_none(float(np.std(finite))) if len(finite) else None,
        }
    marker_means = np.asarray(
        [value["mean"] for value in pcc_per_marker.values() if value["mean"] is not None], dtype=np.float64
    )
    pcc_summary = {
        "aggregation": "mean over slides per marker, then macro mean over markers",
        "mean_of_marker_means": _finite_or_none(float(np.mean(marker_means))) if len(marker_means) else None,
        "per_marker": pcc_per_marker,
    }
    with (output_dir / "cell_pcc_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pcc_summary, handle, indent=2, allow_nan=False)

    classification_markers = cell_cfg.get("classification_markers") or channel_names
    if not isinstance(classification_markers, list):
        raise ValueError("evaluation.cell.classification_markers must be a list")
    missing_labels = [
        marker
        for marker in classification_markers
        if f"label_{marker}" not in valid_rows[0] or f"label_{marker}" not in test_rows[0]
    ]
    if missing_labels:
        raise ValueError(f"Missing cell labels for markers: {missing_labels}")
    feature_markers = cell_cfg.get("feature_markers") or channel_names
    if not isinstance(feature_markers, list) or not all(isinstance(marker, str) for marker in feature_markers):
        raise ValueError("evaluation.cell.feature_markers must be a list")
    unknown_features = sorted(set(feature_markers).difference(channel_names))
    if unknown_features:
        raise ValueError(f"Unknown cell feature markers: {unknown_features}")
    x_train = np.asarray([[row[f"pred_{m}"] for m in feature_markers] for row in valid_rows], dtype=np.float32)
    x_test = np.asarray([[row[f"pred_{m}"] for m in feature_markers] for row in test_rows], dtype=np.float32)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    threshold = float(cell_cfg.get("threshold", 0.5))
    seed = int(cell_cfg.get("seed", 0))
    requested_params = cell_cfg.get("xgb_params", {})
    if not isinstance(requested_params, dict):
        raise ValueError("evaluation.cell.xgb_params must be an object")

    import xgboost as xgb

    class_rows: list[dict[str, Any]] = []
    slide_class_rows: list[dict[str, Any]] = []
    for marker in classification_markers:
        y_train = np.asarray([row[f"label_{marker}"] for row in valid_rows], dtype=np.uint8)
        y_test = np.asarray([row[f"label_{marker}"] for row in test_rows], dtype=np.uint8)
        if len(np.unique(y_train)) < 2:
            continue
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 1,
            "tree_method": "hist",
            "eval_metric": "logloss",
            **requested_params,
        }
        positive = float(np.sum(y_train == 1))
        negative = float(np.sum(y_train == 0))
        params.setdefault("scale_pos_weight", negative / positive if positive else 1.0)
        classifier = xgb.XGBClassifier(**params)
        classifier.fit(x_train, y_train)
        probability = classifier.predict_proba(x_test)[:, 1]
        metrics = _classification_metrics(y_test, probability, threshold)
        class_rows.append(
            {
                "marker": marker,
                "n_valid": len(y_train),
                "n_test": len(y_test),
                **metrics,
            }
        )
        for slide_name in sorted({str(row["slide_name"]) for row in test_rows}):
            indices = np.asarray([i for i, row in enumerate(test_rows) if row["slide_name"] == slide_name])
            slide_metrics = _classification_metrics(y_test[indices], probability[indices], threshold)
            slide_class_rows.append(
                {"slide_name": slide_name, "marker": marker, "n_cells": len(indices), **slide_metrics}
            )

    class_fields = [
        "marker", "n_valid", "n_test", "balanced_accuracy", "f1", "auroc", "tp", "tn", "fp", "fn"
    ]
    _write_csv(output_dir / "cell_classification_per_marker.csv", class_rows, class_fields)
    _write_csv(
        output_dir / "cell_classification_per_slide_marker.csv",
        slide_class_rows,
        ["slide_name", "marker", "n_cells", "balanced_accuracy", "f1", "auroc", "tp", "tn", "fp", "fn"],
    )
    balanced_accuracy_values = [float(row["balanced_accuracy"]) for row in class_rows]
    f1_values = [float(row["f1"]) for row in class_rows]
    auroc_values = np.asarray([float(row["auroc"]) for row in class_rows], dtype=np.float64)
    finite_auroc = auroc_values[np.isfinite(auroc_values)]
    class_summary = {
        "protocol": "standardized predicted marker features; marker-specific XGBoost trained on valid cells and evaluated on test cells",
        "threshold": threshold,
        "feature_markers": feature_markers,
        "n_valid_cells": len(valid_rows),
        "n_test_cells": len(test_rows),
        "n_markers": len(class_rows),
        "mean_balanced_accuracy": (
            _finite_or_none(float(np.mean(balanced_accuracy_values))) if balanced_accuracy_values else None
        ),
        "mean_f1": _finite_or_none(float(np.mean(f1_values))) if f1_values else None,
        "mean_auroc": _finite_or_none(float(np.mean(finite_auroc))) if len(finite_auroc) else None,
    }
    with (output_dir / "cell_classification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(class_summary, handle, indent=2, allow_nan=False)

    return {
        "enabled": True,
        "annotations_csv": str(annotations_path),
        "valid_pred_dir": str(valid_pred),
        "test_pred_dir": str(test_pred),
        "n_valid_cells": len(valid_rows),
        "n_test_cells": len(test_rows),
        "pcc": pcc_summary,
        "classification": class_summary,
        "output_dir": str(output_dir),
    }
