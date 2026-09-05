from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .clinical_features import align_he_features, extract_virtual_features
from .clinical_metrics import (
    aggregate_patient_predictions,
    assign_median_risk_groups,
    kaplan_meier_rows,
    logrank_test,
    patient_cluster_bootstrap,
    survival_fold_metrics,
)
from .clinical_models import build_clinical_model, discrete_time_nll, forward_clinical_model
from .experiment import load_experiment_config, parse_overrides, set_dotted


SCHEMA_VERSION = "heprobench_clinical_v1"
STAGES = (
    "prepare-outcomes",
    "extract-features",
    "align-features",
    "make-splits",
    "train",
    "infer",
    "evaluate",
)


def _path(value: str | Path, source_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _resolve_config(
    config_path: str | Path,
    *,
    device: str | None,
    output_dir: str | Path | None,
    overrides: list[str] | None,
) -> tuple[dict[str, Any], Path, Path]:
    config, source_path = load_experiment_config(config_path)
    config = copy.deepcopy(config)
    if config.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"Clinical config schema_version must be {SCHEMA_VERSION!r}")
    config["schema_version"] = SCHEMA_VERSION
    if device is not None:
        config.setdefault("runtime", {})["device"] = device
    if output_dir is not None:
        config.setdefault("output", {})["run_dir"] = str(output_dir)
    for key, value in parse_overrides(overrides):
        set_dotted(config, key, value)
    for required in ("clinical", "data", "features", "model", "train", "evaluation"):
        if not isinstance(config.get(required), dict):
            raise ValueError(f"Clinical config requires an object named {required!r}")
    task_type = str(config["clinical"].get("task", "survival")).lower()
    if task_type not in {"survival", "classification"}:
        raise ValueError("clinical.task must be 'survival' or 'classification'")
    run_dir_value = config.get("output", {}).get("run_dir")
    run_dir = (
        _path(run_dir_value, source_path)
        if run_dir_value
        else (Path.cwd() / "outputs" / "clinical" / source_path.stem).resolve()
    )
    metadata = config.setdefault("_heprobench", {})
    metadata.update(
        {
            "source_config": str(source_path),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    metadata["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config, source_path, run_dir


def _expand_stages(config: dict[str, Any], stages: list[str]) -> list[str]:
    requested = stages or ["all"]
    output: list[str] = []
    for stage in requested:
        normalized = stage.strip().lower()
        if normalized == "all":
            configured = config.get("pipeline", {}).get("stages", list(STAGES))
            if not isinstance(configured, list):
                raise ValueError("pipeline.stages must be a list")
            output.extend(str(item).lower() for item in configured)
        else:
            output.append(normalized)
    output = list(dict.fromkeys(output))
    unknown = sorted(set(output).difference(STAGES))
    if unknown:
        raise ValueError(f"Unknown clinical stages: {unknown}; expected one of {list(STAGES)} or 'all'")
    return output


def prepare_outcomes(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    data_cfg = config["data"]
    columns = config["clinical"].get("columns", {})
    raw_path = _path(data_cfg["raw_clinical_csv"], source_path)
    manifest_path = _path(data_cfg["manifest_csv"], source_path)
    frame = pd.read_csv(raw_path)
    patient_column = str(columns.get("raw_patient_id", "Patient ID"))
    output = pd.DataFrame({"patient_id": frame[patient_column].astype(str)})
    task_type = str(config["clinical"].get("task", "survival")).lower()
    if task_type == "survival":
        time_column = str(columns.get("raw_event_time", "Overall Survival (Months)"))
        status_column = str(columns.get("raw_status", "Overall Survival Status"))
        output["event_time"] = pd.to_numeric(frame[time_column], errors="coerce")
        output["censor"] = frame[status_column].map(_normalize_censor_status)
        output = output.dropna(subset=["event_time", "censor"])
        output = output[output["event_time"] >= 0]
    else:
        label_column = str(columns.get("raw_label", columns.get("label", "Class")))
        output["label"] = frame[label_column]
        output = output.dropna(subset=["label"])
    output["slide_id"] = output["patient_id"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(manifest_path, index=False)
    return {
        "stage": "prepare-outcomes",
        "input": str(raw_path),
        "output": str(manifest_path),
        "n_rows": len(output),
        "n_patients": int(output["patient_id"].nunique()),
    }


def _normalize_censor_status(value: Any) -> float:
    normalized = str(value).strip().lower()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1].strip()
    if normalized in {"deceased", "dead", "died", "0", "0.0"}:
        return 0.0
    if normalized in {"living", "alive", "1", "1.0"}:
        return 1.0
    return float("nan")


def _load_manifest(config: dict[str, Any], source_path: Path) -> pd.DataFrame:
    path = _path(config["data"]["manifest_csv"], source_path)
    frame = pd.read_csv(path)
    columns = config["clinical"].get("columns", {})
    rename = {
        str(columns.get("patient_id", "patient_id")): "patient_id",
        str(columns.get("slide_id", "slide_id")): "slide_id",
    }
    task_type = str(config["clinical"].get("task", "survival")).lower()
    if task_type == "survival":
        rename.update(
            {
                str(columns.get("event_time", "event_time")): "event_time",
                str(columns.get("censor", "censor")): "censor",
            }
        )
    else:
        rename[str(columns.get("label", "label"))] = "label"
    frame = frame.rename(columns=rename)
    required = {"patient_id"}
    required.update({"event_time", "censor"} if task_type == "survival" else {"label"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Clinical manifest is missing columns: {missing}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if "slide_id" not in frame.columns:
        frame["slide_id"] = frame["patient_id"]
    frame["slide_id"] = frame["slide_id"].astype(str)
    if task_type == "survival":
        frame["event_time"] = pd.to_numeric(frame["event_time"], errors="raise")
        frame["censor"] = pd.to_numeric(frame["censor"], errors="raise")
        if not frame["censor"].isin([0, 1]).all():
            raise ValueError("Clinical censor column must use 1=censored, 0=event observed")
    _validate_patient_labels(frame, task_type)
    return frame.reset_index(drop=True)


def _validate_patient_labels(frame: pd.DataFrame, task_type: str) -> None:
    label_columns = ["event_time", "censor"] if task_type == "survival" else ["label"]
    uniqueness = frame.groupby("patient_id")[label_columns].nunique(dropna=False)
    if (uniqueness > 1).any().any():
        raise ValueError("Clinical labels must be consistent across slides belonging to one patient")


def _map_slide_ids(frame: pd.DataFrame, feature_root: Path, policy: str) -> pd.DataFrame:
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Clinical feature root not found: {feature_root}")
    slide_dirs = sorted(path.name for path in feature_root.iterdir() if path.is_dir())
    resolved: list[str] = []
    for _, row in frame.iterrows():
        requested = str(row["slide_id"]).strip()
        if requested in slide_dirs:
            resolved.append(requested)
            continue
        patient_id = str(row["patient_id"]).strip()
        matches = [item for item in slide_dirs if item.startswith(requested or patient_id)]
        if not matches and requested != patient_id:
            matches = [item for item in slide_dirs if item.startswith(patient_id)]
        if not matches:
            if policy == "drop":
                resolved.append("")
                continue
            raise FileNotFoundError(f"No feature directory matches slide/patient {requested or patient_id!r}")
        if len(matches) > 1 and policy == "error":
            raise ValueError(f"Multiple feature directories match {requested or patient_id!r}: {matches}")
        resolved.append(matches[0])
    result = frame.copy()
    result["slide_id"] = resolved
    return result[result["slide_id"].str.len() > 0].reset_index(drop=True)


def make_splits(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    frame = _load_manifest(config, source_path)
    feature_root = _path(config["features"]["root"], source_path)
    mapping_policy = str(config.get("split", {}).get("unmatched_policy", "error")).lower()
    if mapping_policy not in {"error", "first", "drop"}:
        raise ValueError("split.unmatched_policy must be 'error', 'first', or 'drop'")
    frame = _map_slide_ids(frame, feature_root, mapping_policy)

    manifest_path = _path(config["data"]["manifest_csv"], source_path)
    frame.to_csv(manifest_path, index=False)
    split_cfg = config.get("split", {})
    n_folds = int(split_cfg.get("folds", 5))
    seed = int(split_cfg.get("seed", 1))
    if n_folds < 2:
        raise ValueError("split.folds must be at least 2")
    patients = frame.drop_duplicates("patient_id").reset_index(drop=True)
    task_type = str(config["clinical"].get("task", "survival")).lower()
    stratify_column = split_cfg.get("stratify_column")
    if stratify_column:
        stratify_column = str(stratify_column)
        if stratify_column not in patients.columns:
            if task_type == "classification" and stratify_column == "label":
                pass
            else:
                raise ValueError(f"split.stratify_column not found: {stratify_column}")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.arange(len(patients)), patients[stratify_column]))
    else:
        rng = np.random.RandomState(seed)
        shuffled = np.arange(len(patients))
        rng.shuffle(shuffled)
        sizes = [len(patients) // n_folds + (1 if index < len(patients) % n_folds else 0) for index in range(n_folds)]
        folds = []
        start = 0
        for size in sizes:
            validation = shuffled[start : start + size]
            training = np.setdiff1d(shuffled, validation, assume_unique=False)
            folds.append((training, validation))
            start += size

    split_dir = _path(config["data"]["split_dir"], source_path)
    split_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(folds):
        train_patients = set(patients.iloc[train_index]["patient_id"].astype(str))
        validation_patients = set(patients.iloc[validation_index]["patient_id"].astype(str))
        if train_patients.intersection(validation_patients):
            raise AssertionError("Patient leakage detected while creating clinical folds")
        train_ids = frame[frame["patient_id"].isin(train_patients)]["slide_id"].astype(str).tolist()
        validation_ids = frame[frame["patient_id"].isin(validation_patients)]["slide_id"].astype(str).tolist()
        maximum = max(len(train_ids), len(validation_ids))
        rows = [
            {
                "train": train_ids[index] if index < len(train_ids) else "",
                "val": validation_ids[index] if index < len(validation_ids) else "",
            }
            for index in range(maximum)
        ]
        _write_csv(split_dir / f"splits_{fold}.csv", rows)
        fold_summaries.append(
            {
                "fold": fold,
                "n_train_slides": len(train_ids),
                "n_val_slides": len(validation_ids),
                "n_train_patients": len(train_patients),
                "n_val_patients": len(validation_patients),
            }
        )
    summary = {
        "stage": "make-splits",
        "split_dir": str(split_dir),
        "seed": seed,
        "patient_level": True,
        "folds": fold_summaries,
    }
    _write_json(split_dir / "split_summary.json", summary)
    return summary


def _read_split(path: Path) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(path)
    if not {"train", "val"}.issubset(frame.columns):
        raise ValueError(f"Split must contain train and val columns: {path}")
    return (
        frame["train"].dropna().astype(str).tolist(),
        frame["val"].dropna().astype(str).tolist(),
    )


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32, copy=False)
        return torch.from_numpy(array)
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Feature file must contain one tensor: {path}")
        return value.detach().float().cpu()
    raise ValueError(f"Unsupported clinical feature file: {path}")


def _feature_files(config: dict[str, Any], source_path: Path, slide_id: str) -> tuple[Path, Path]:
    feature_cfg = config["features"]
    slide_dir = _path(feature_cfg["root"], source_path) / slide_id
    he_path = slide_dir / str(feature_cfg.get("he_aligned_file", "he_aligned.npy"))
    representation = str(config["model"].get("virtual_representation", "channels")).lower()
    if representation == "channels":
        virtual_name = str(feature_cfg.get("virtual_channels_file", "virtual_channels.npy"))
    elif representation == "aggregated":
        virtual_name = str(feature_cfg.get("virtual_aggregated_file", "virtual_aggregated.npy"))
    else:
        raise ValueError("model.virtual_representation must be 'channels' or 'aggregated'")
    return he_path, slide_dir / virtual_name


def _load_bag(
    config: dict[str, Any], source_path: Path, row: pd.Series
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    modality = str(config["model"].get("input_modality", "virtual")).lower()
    if modality not in {"he", "virtual", "both"}:
        raise ValueError("model.input_modality must be 'he', 'virtual', or 'both'")
    he_path, virtual_path = _feature_files(config, source_path, str(row["slide_id"]))
    he = _load_tensor(he_path) if modality in {"he", "both"} else None
    virtual = _load_tensor(virtual_path) if modality in {"virtual", "both"} else None
    if he is not None and he.ndim != 2:
        raise ValueError(f"H&E bag must have shape [patch,feature], got {tuple(he.shape)}")
    if virtual is not None and virtual.ndim not in {2, 3}:
        raise ValueError(f"Virtual bag must have shape [patch,feature] or [patch,channel,feature], got {tuple(virtual.shape)}")
    if he is not None and virtual is not None and he.shape[0] != virtual.shape[0]:
        raise ValueError(f"Unaligned clinical bags for slide {row['slide_id']}")
    return he, virtual


def _time_bins(frame: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    times = frame["event_time"].to_numpy(dtype=np.float32)
    edges = np.quantile(times, np.linspace(0.0, 1.0, n_bins + 1))
    if np.unique(edges).size < n_bins + 1:
        edges = np.linspace(float(times.min()), float(times.max()), n_bins + 1)
    labels = np.digitize(times, edges[1:-1], right=False).astype(np.int64)
    return labels, edges.astype(np.float32)


def _task_metadata(config: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    task_type = str(config["clinical"].get("task", "survival")).lower()
    if task_type == "survival":
        n_outputs = int(config["model"].get("n_time_bins", 4))
        labels, bins = _time_bins(frame, n_outputs)
        frame["survival_bin"] = labels
        return {"task_type": task_type, "output_dim": n_outputs, "time_bins": bins.tolist()}
    class_names_cfg = config["clinical"].get("class_names")
    class_names = list(class_names_cfg) if class_names_cfg else sorted(frame["label"].astype(str).unique())
    label_to_index = {str(name): index for index, name in enumerate(class_names)}
    if not set(frame["label"].astype(str)).issubset(label_to_index):
        raise ValueError("clinical.class_names does not cover all manifest labels")
    frame["class_index"] = frame["label"].astype(str).map(label_to_index).astype(int)
    return {
        "task_type": task_type,
        "output_dim": len(class_names),
        "class_names": class_names,
    }


def _model_dimensions(
    config: dict[str, Any], source_path: Path, frame: pd.DataFrame
) -> tuple[int | None, int | None]:
    he, virtual = _load_bag(config, source_path, frame.iloc[0])
    he_dim = int(he.shape[-1]) if he is not None else None
    virtual_dim = int(virtual.shape[-1]) if virtual is not None else None
    expected_he = config["model"].get("he_input_dim")
    expected_virtual = config["model"].get("virtual_input_dim")
    if expected_he is not None and he_dim is not None and int(expected_he) != he_dim:
        raise ValueError(f"Configured H&E dimension {expected_he} != observed {he_dim}")
    if expected_virtual is not None and virtual_dim is not None and int(expected_virtual) != virtual_dim:
        raise ValueError(f"Configured virtual dimension {expected_virtual} != observed {virtual_dim}")
    return he_dim, virtual_dim


def _device(config: dict[str, Any]) -> torch.device:
    configured_value = config.get("runtime", {}).get("device")
    if configured_value is None:
        configured_value = "cuda" if torch.cuda.is_available() else "cpu"
    configured = str(configured_value)
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Clinical config requested {configured}, but CUDA is unavailable")
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _subset(frame: pd.DataFrame, slide_ids: Iterable[str]) -> pd.DataFrame:
    order = {slide_id: index for index, slide_id in enumerate(slide_ids)}
    selected = frame[frame["slide_id"].astype(str).isin(order)].copy()
    selected["_order"] = selected["slide_id"].astype(str).map(order)
    return selected.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _classification_values(rows: list[dict[str, Any]], class_names: list[str]) -> dict[str, float]:
    truth = np.asarray([int(row["class_index"]) for row in rows], dtype=int)
    probabilities = np.asarray(
        [[float(row[f"probability_{index}"]) for index in range(len(class_names))] for row in rows],
        dtype=np.float64,
    )
    prediction = probabilities.argmax(axis=1)
    try:
        if len(class_names) == 2:
            auc = float(roc_auc_score(truth, probabilities[:, 1]))
        else:
            auc = float(roc_auc_score(truth, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def _evaluate_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    source_path: Path,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for _, item in frame.iterrows():
            he, virtual = _load_bag(config, source_path, item)
            he = he.to(device) if he is not None else None
            virtual = virtual.to(device) if virtual is not None else None
            payload = forward_clinical_model(model, config["model"], he=he, virtual=virtual)
            row: dict[str, Any] = {
                "patient_id": str(item["patient_id"]),
                "slide_id": str(item["slide_id"]),
            }
            if metadata["task_type"] == "survival":
                label = torch.tensor([int(item["survival_bin"])], device=device)
                censor = torch.tensor([float(item["censor"])], device=device)
                loss = discrete_time_nll(
                    payload["hazards"],
                    payload["survival"],
                    label,
                    censor,
                    alpha=float(config["train"].get("alpha_survival", 0.0)),
                )
                row.update(
                    {
                        "event_time": float(item["event_time"]),
                        "censor": float(item["censor"]),
                        "event_observed": int(1 - int(item["censor"])),
                        "survival_bin": int(item["survival_bin"]),
                        "risk": float(-payload["survival"].sum(dim=1).item()),
                        "hazards": json.dumps(payload["hazards"].squeeze(0).cpu().tolist()),
                    }
                )
            else:
                label = torch.tensor([int(item["class_index"])], device=device)
                loss = torch.nn.functional.cross_entropy(payload["logits"], label)
                probabilities = torch.softmax(payload["logits"], dim=1).squeeze(0).cpu().tolist()
                row.update(
                    {
                        "label": str(item["label"]),
                        "class_index": int(item["class_index"]),
                        **{f"probability_{index}": float(value) for index, value in enumerate(probabilities)},
                    }
                )
            losses.append(float(loss.item()))
            rows.append(row)
    return float(np.mean(losses)), rows


def train_clinical(
    config: dict[str, Any], source_path: Path, run_dir: Path
) -> dict[str, Any]:
    if int(config["train"].get("batch_size", 1)) != 1:
        raise ValueError("Clinical variable-length bags currently require train.batch_size=1")
    frame = _load_manifest(config, source_path)
    metadata = _task_metadata(config, frame)
    he_dim, virtual_dim = _model_dimensions(config, source_path, frame)
    device = _device(config)
    split_dir = _path(config["data"]["split_dir"], source_path)
    n_folds = int(config.get("split", {}).get("folds", 5))
    epochs = int(config["train"].get("epochs", 100))
    gradient_accumulation = int(config["train"].get("gradient_accumulation", 32))
    learning_rate = float(config["train"].get("learning_rate", 2e-4))
    weight_decay = float(config["train"].get("weight_decay", 1e-5))
    seed = int(config["train"].get("seed", config.get("split", {}).get("seed", 1)))
    checkpoint_dir = run_dir / "checkpoints"
    history_dir = run_dir / "history"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []

    for fold in range(n_folds):
        train_ids, validation_ids = _read_split(split_dir / f"splits_{fold}.csv")
        train_frame = _subset(frame, train_ids)
        validation_frame = _subset(frame, validation_ids)
        if train_frame.empty or validation_frame.empty:
            raise ValueError(f"Clinical fold {fold} has an empty train or validation subset")
        _seed_everything(seed + fold, device)
        model = build_clinical_model(
            config["model"],
            task_type=metadata["task_type"],
            he_input_dim=he_dim,
            virtual_input_dim=virtual_dim,
            output_dim=int(metadata["output_dim"]),
        ).to(device)
        optimizer_name = str(config["train"].get("optimizer", "adam")).lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise ValueError("train.optimizer must be 'adam' or 'sgd'")

        best_loss = float("inf")
        best_epoch = -1
        history: list[dict[str, Any]] = []
        checkpoint_path = checkpoint_dir / f"fold_{fold}_best.pt"
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rng = np.random.default_rng(seed + fold * 100_000 + epoch)
            indices = rng.permutation(len(train_frame))
            train_losses: list[float] = []
            for position, index in enumerate(indices):
                item = train_frame.iloc[int(index)]
                he, virtual = _load_bag(config, source_path, item)
                he = he.to(device) if he is not None else None
                virtual = virtual.to(device) if virtual is not None else None
                payload = forward_clinical_model(model, config["model"], he=he, virtual=virtual)
                if metadata["task_type"] == "survival":
                    loss = discrete_time_nll(
                        payload["hazards"],
                        payload["survival"],
                        torch.tensor([int(item["survival_bin"])], device=device),
                        torch.tensor([float(item["censor"])], device=device),
                        alpha=float(config["train"].get("alpha_survival", 0.0)),
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        payload["logits"],
                        torch.tensor([int(item["class_index"])], device=device),
                    )
                train_losses.append(float(loss.item()))
                (loss / gradient_accumulation).backward()
                if (position + 1) % gradient_accumulation == 0 or position + 1 == len(indices):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            validation_loss, prediction_rows = _evaluate_model(
                model, config, source_path, validation_frame, metadata, device
            )
            epoch_row: dict[str, Any] = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
            if metadata["task_type"] == "survival":
                fold_metrics, _ = survival_fold_metrics(
                    [{**row, "fold": fold} for row in prediction_rows]
                )
                epoch_row["validation_c_index"] = fold_metrics[0]["c_index"]
            else:
                epoch_row.update(
                    {f"validation_{key}": value for key, value in _classification_values(
                        prediction_rows, metadata["class_names"]
                    ).items()}
                )
            history.append(epoch_row)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                torch.save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "fold": fold,
                        "model_state": model.state_dict(),
                        "model": config["model"],
                        "metadata": metadata,
                        "he_input_dim": he_dim,
                        "virtual_input_dim": virtual_dim,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_loss,
                        "config_sha256": config["_heprobench"]["config_sha256"],
                    },
                    checkpoint_path,
                )
        _write_csv(history_dir / f"fold_{fold}.csv", history)
        fold_summaries.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "checkpoint": str(checkpoint_path),
                "n_train": len(train_frame),
                "n_validation": len(validation_frame),
            }
        )
    summary = {
        "stage": "train",
        "task": metadata["task_type"],
        "model": config["model"].get("name", "amil"),
        "device": str(device),
        "folds": fold_summaries,
    }
    _write_json(run_dir / "training_summary.json", summary)
    return summary


def infer_clinical(config: dict[str, Any], source_path: Path, run_dir: Path) -> dict[str, Any]:
    frame = _load_manifest(config, source_path)
    metadata = _task_metadata(config, frame)
    device = _device(config)
    split_dir = _path(config["data"]["split_dir"], source_path)
    n_folds = int(config.get("split", {}).get("folds", 5))
    inference_split = str(config.get("inference", {}).get("split", "val")).lower()
    if inference_split not in {"val", "train", "all"}:
        raise ValueError("inference.split must be 'val', 'train', or 'all'")
    rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        checkpoint_path = run_dir / "checkpoints" / f"fold_{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_clinical_model(
            checkpoint["model"],
            task_type=metadata["task_type"],
            he_input_dim=checkpoint.get("he_input_dim"),
            virtual_input_dim=checkpoint.get("virtual_input_dim"),
            output_dim=int(metadata["output_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        if inference_split == "all":
            selected = frame
        else:
            train_ids, validation_ids = _read_split(split_dir / f"splits_{fold}.csv")
            selected = _subset(frame, validation_ids if inference_split == "val" else train_ids)
        _, fold_rows = _evaluate_model(model, config, source_path, selected, metadata, device)
        rows.extend({"fold": fold, "split": inference_split, **row} for row in fold_rows)
    prediction_path = run_dir / "clinical_predictions.csv"
    _write_csv(prediction_path, rows)
    summary = {
        "stage": "infer",
        "task": metadata["task_type"],
        "split": inference_split,
        "n_predictions": len(rows),
        "n_patients": len({(row["fold"], row["patient_id"]) for row in rows}),
        "output": str(prediction_path),
    }
    _write_json(run_dir / "inference_summary.json", summary)
    return summary


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_km(path: Path, km_rows: list[dict[str, Any]], p_value: float) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    for group in ("low", "high"):
        selected = [row for row in km_rows if row["risk_group"] == group]
        if selected:
            axis.step(
                [float(row["time"]) for row in selected],
                [float(row["survival_probability"]) for row in selected],
                where="post",
                label=f"{group.capitalize()} risk",
            )
    axis.set(xlabel="Time", ylabel="Survival probability", ylim=(0, 1.02))
    axis.text(0.98, 0.04, f"log-rank p={p_value:.3g}", transform=axis.transAxes, ha="right")
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return True


def _cox_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return {"enabled": False, "reason": "lifelines is not installed"}
    frame = pd.DataFrame(rows)
    risk_std = float(frame["risk"].astype(float).std(ddof=0))
    if risk_std == 0:
        return {"enabled": False, "reason": "risk is constant"}
    frame["risk_z"] = (frame["risk"].astype(float) - frame["risk"].astype(float).mean()) / risk_std
    frame["event_observed"] = frame["event_observed"].astype(int)
    frame["event_time"] = frame["event_time"].astype(float)
    fitter = CoxPHFitter()
    fitter.fit(frame[["event_time", "event_observed", "risk_z"]], "event_time", "event_observed")
    result = fitter.summary.loc["risk_z"]
    return {
        "enabled": True,
        "covariate": "risk_z",
        "hazard_ratio": float(result["exp(coef)"]),
        "ci_lower": float(result["exp(coef) lower 95%"]),
        "ci_upper": float(result["exp(coef) upper 95%"]),
        "p_value": float(result["p"]),
    }


def evaluate_clinical(config: dict[str, Any], source_path: Path, run_dir: Path) -> dict[str, Any]:
    prediction_path = run_dir / "clinical_predictions.csv"
    rows = _read_prediction_rows(prediction_path)
    if not rows:
        raise ValueError(f"Clinical prediction file is empty: {prediction_path}")
    task_type = str(config["clinical"].get("task", "survival")).lower()
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    if task_type == "classification":
        class_names = list(config["clinical"].get("class_names") or sorted({row["label"] for row in rows}))
        fold_rows = []
        for fold in sorted({int(row["fold"]) for row in rows}):
            selected = [row for row in rows if int(row["fold"]) == fold]
            fold_rows.append({"fold": fold, "n_samples": len(selected), **_classification_values(selected, class_names)})
        patient_rows = aggregate_patient_predictions(rows)
        patient_metrics = _classification_values(patient_rows, class_names)
        _write_csv(evaluation_dir / "classification_per_fold.csv", fold_rows)
        _write_csv(evaluation_dir / "patient_predictions.csv", patient_rows)
        summary = {
            "stage": "evaluate",
            "task": task_type,
            "class_names": class_names,
            "fold_macro": {
                key: float(np.nanmean([float(row[key]) for row in fold_rows]))
                for key in ("auc", "accuracy", "macro_f1")
            },
            "patient": patient_metrics,
        }
    else:
        slide_fold_rows, slide_summary = survival_fold_metrics(rows)
        patient_rows = aggregate_patient_predictions(rows)
        patient_fold_rows, patient_summary = survival_fold_metrics(patient_rows)
        evaluation_cfg = config["evaluation"]
        bootstrap = patient_cluster_bootstrap(
            patient_rows,
            iterations=int(evaluation_cfg.get("bootstrap_iterations", 1000)),
            seed=int(evaluation_cfg.get("bootstrap_seed", 3407)),
            confidence_level=float(evaluation_cfg.get("confidence_level", 0.95)),
        )
        grouped_rows = assign_median_risk_groups(patient_rows)
        km_rows = kaplan_meier_rows(grouped_rows)
        logrank = logrank_test(grouped_rows)
        _write_csv(evaluation_dir / "survival_slide_per_fold.csv", slide_fold_rows)
        _write_csv(evaluation_dir / "survival_patient_per_fold.csv", patient_fold_rows)
        _write_csv(evaluation_dir / "patient_risk_predictions.csv", grouped_rows)
        _write_csv(evaluation_dir / "kaplan_meier.csv", km_rows)
        plot_written = False
        if bool(evaluation_cfg.get("plot_km", True)):
            plot_written = _plot_km(evaluation_dir / "kaplan_meier.svg", km_rows, logrank["p_value"])
        cox = _cox_summary(grouped_rows) if bool(evaluation_cfg.get("cox", True)) else {"enabled": False}
        summary = {
            "stage": "evaluate",
            "task": task_type,
            "primary_level": "patient",
            "slide": slide_summary,
            "patient": patient_summary,
            "bootstrap": bootstrap,
            "kaplan_meier": {
                "grouping": "fold_specific_median",
                "logrank": logrank,
                "plot_written": plot_written,
            },
            "cox": cox,
        }
    _write_json(evaluation_dir / "summary.json", summary)
    return summary


def run_clinical_pipeline(
    config_path: str | Path,
    *,
    stages: list[str] | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
    overrides: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config, source_path, run_dir = _resolve_config(
        config_path,
        device=device,
        output_dir=output_dir,
        overrides=overrides,
    )
    selected_stages = _expand_stages(config, stages or ["all"])
    plan = {
        "schema_version": SCHEMA_VERSION,
        "config": str(source_path),
        "run_dir": str(run_dir),
        "task": config["clinical"].get("task", "survival"),
        "model": config["model"].get("name", "amil"),
        "input_modality": config["model"].get("input_modality", "virtual"),
        "stages": selected_stages,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return plan
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "resolved_config.clinical.json", config)
    results: list[dict[str, Any]] = []
    for stage in selected_stages:
        if stage == "prepare-outcomes":
            result = prepare_outcomes(config, source_path)
        elif stage == "extract-features":
            result = extract_virtual_features(config, source_path)
        elif stage == "align-features":
            result = align_he_features(config, source_path)
        elif stage == "make-splits":
            result = make_splits(config, source_path)
        elif stage == "train":
            result = train_clinical(config, source_path, run_dir)
        elif stage == "infer":
            result = infer_clinical(config, source_path, run_dir)
        elif stage == "evaluate":
            result = evaluate_clinical(config, source_path, run_dir)
        else:
            raise AssertionError(stage)
        results.append(result)
    payload = {**plan, "results": results}
    _write_json(run_dir / "clinical_pipeline_summary.json", payload)
    return payload
