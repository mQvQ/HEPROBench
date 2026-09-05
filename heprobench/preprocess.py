from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .experiment import load_experiment_config, parse_overrides, set_dotted


SCHEMA_VERSION = "heprobench_preprocessing_v1"
STAGES = (
    "split",
    "register",
    "registration-qc",
    "tile",
    "normalize",
    "patch-qc",
    "segment",
    "cell-extract",
    "gate",
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.\[\],+-]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    return path


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _safe_id(value: Any, description: str) -> str:
    identifier = str(value).strip()
    if not identifier or not _SAFE_ID.fullmatch(identifier) or identifier in {".", ".."}:
        raise ValueError(
            f"{description} must be a non-empty filesystem-safe identifier "
            "using letters, numbers, '_', '-', '.', '[', ']', ',', or '+': "
            f"{identifier!r}"
        )
    return identifier


def _canonical_identifiers(ctx: dict[str, Any], row: dict[str, str]) -> tuple[str, str, str]:
    columns = ctx["config"]["dataset"].get("columns", {})
    slide_col = str(columns.get("slide_id", "slide_id"))
    sample_col = str(columns.get("sample_id", slide_col))
    fov_col = str(columns.get("fov_id", sample_col))
    sample_id = _safe_id(row.get(sample_col, ""), "sample_id")
    slide_name = _safe_id(row.get(slide_col, ""), "slide_id")
    fov_name = _safe_id(row.get(fov_col, ""), "fov_id")
    return sample_id, slide_name, fov_name


def _as_nonnegative_int(value: Any, description: str) -> int:
    text = str(value).strip()
    number = 0 if text == "" else int(text)
    if number < 0:
        raise ValueError(f"{description} must be non-negative, received {number}")
    return number


def _stage_report(
    ctx: dict[str, Any], stage: str, *, inputs: list[Path], outputs: list[Path], details: dict[str, Any]
) -> dict[str, Any]:
    report = {
        "schema": "heprobench_preprocessing_stage_v1",
        "dataset": ctx["config"]["dataset"]["name"],
        "stage": stage,
        "created_at": _utc_now(),
        "config_sha256": ctx["config_sha256"],
        "inputs": [
            {"path": str(path), "sha256": _sha256(path) if path.is_file() else None} for path in inputs
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path) if path.is_file() else None} for path in outputs
        ],
        **details,
    }
    destination = ctx["reports_dir"] / f"{stage}.json"
    _write_json(destination, report)
    return report


def load_preprocessing_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    config, source = load_experiment_config(config_path)
    for key, value in parse_overrides(overrides):
        set_dotted(config, key, value)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r} in {source}")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("name"):
        raise ValueError("preprocessing config requires dataset.name")
    source_dir = source.parent
    data_root = _resolve(dataset.get("root_dir", "."), source_dir)
    output = config.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("preprocessing output must be an object")
    requested_output = output_dir or output.get("root_dir", "../../outputs/preprocessing")
    resolved_output = _resolve(requested_output, source_dir)
    manifest = _resolve(dataset.get("slide_manifest", "manifests/slides.csv"), data_root)
    channels_file = _resolve(dataset["channel_names_file"], source_dir)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "config": config,
        "source": source,
        "data_root": data_root,
        "output_root": resolved_output,
        "manifest": manifest,
        "channels_file": channels_file,
        "reports_dir": resolved_output / "reports",
        "config_sha256": config_sha,
    }


def _channels(ctx: dict[str, Any]) -> list[str]:
    with _require_file(ctx["channels_file"], "retained channel-name file").open(encoding="utf-8") as handle:
        channels = json.load(handle)
    if not isinstance(channels, list) or not channels or not all(isinstance(item, str) for item in channels):
        raise ValueError("channel_names_file must contain a non-empty JSON string list")
    if len(set(channels)) != len(channels):
        raise ValueError("retained channel names must be unique")
    return channels


def _validate_preprocessing_config(ctx: dict[str, Any]) -> list[str]:
    config = ctx["config"]
    dataset = config["dataset"]
    channels = _channels(ctx)
    declared_count = dataset.get("channel_count")
    if declared_count is not None and int(declared_count) != len(channels):
        raise ValueError(
            f"dataset.channel_count={declared_count} does not match {len(channels)} channel names"
        )
    access = dataset.get("access", {})
    if access and access.get("data_in_repository") is not False:
        raise ValueError("Dataset preprocessing configs must set access.data_in_repository=false")

    tiling = config.get("tiling", {})
    if tiling:
        raw_count = int(tiling.get("raw_channel_count", len(channels)))
        indices = [int(value) for value in tiling.get("retained_channel_indices", range(raw_count))]
        if len(indices) != len(channels):
            raise ValueError("tiling.retained_channel_indices must match channel_names_file length")
        if not indices or min(indices) < 0 or max(indices) >= raw_count:
            raise ValueError("tiling.retained_channel_indices must be within tiling.raw_channel_count")
        patch_size = int(tiling.get("patch_size", 256))
        fov_size = tiling.get("fov_size_px")
        if fov_size is not None and (int(fov_size) < patch_size or int(fov_size) % patch_size):
            raise ValueError("tiling.fov_size_px must be null or a positive multiple of patch_size")

    registration = config.get("registration", {})
    method = str(registration.get("method", "valis")).lower().replace("-", "_")
    if method not in {"valis", "identity", "pre_registered"}:
        raise ValueError("registration.method must be one of: valis, identity, pre_registered")

    registration_qc = config.get("registration_qc", {})
    if registration_qc.get("enabled", True) and tiling:
        raw_index = int(registration_qc.get("raw_nuclear_channel_index", 0))
        if raw_index < 0 or raw_index >= int(tiling.get("raw_channel_count", len(channels))):
            raise ValueError("registration_qc.raw_nuclear_channel_index is outside the raw channel panel")

    patch_qc = config.get("patch_qc", {})
    if patch_qc.get("enabled", True) and patch_qc:
        nuclear = str(patch_qc.get("nuclear_channel", ""))
        if nuclear not in channels:
            raise ValueError(f"Unknown patch_qc.nuclear_channel: {nuclear!r}")

    segmentation = config.get("segmentation", {})
    if segmentation:
        required = [str(segmentation.get("nuclear_channel", "")), *segmentation.get("boundary_channels", [])]
        unknown = sorted({name for name in required if name not in channels})
        if unknown:
            raise ValueError(f"Unknown segmentation channels: {unknown}")
        if not segmentation.get("boundary_channels"):
            raise ValueError("segmentation.boundary_channels must contain at least one marker")

    warnings: list[str] = []
    reproduction = config.get("reproduction", {})
    if reproduction.get("status") not in {"verified", "partially_verified", "pending_source_audit"}:
        warnings.append("reproduction.status is not declared")
    return warnings


def _raw_path(ctx: dict[str, Any], value: str) -> Path:
    return _resolve(value, ctx["data_root"])


def _resolved_config_payload(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        **ctx["config"],
        "_heprobench": {
            "source_config": str(ctx["source"]),
            "data_root": str(ctx["data_root"]),
            "output_root": str(ctx["output_root"]),
            "config_sha256": ctx["config_sha256"],
            "resolved_at": _utc_now(),
        },
    }


def _split(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["manifest"], "dataset sample manifest")
    rows = _read_csv(source)
    if not rows:
        raise ValueError(f"Slide manifest contains no records: {source}")
    cfg = ctx["config"].get("split", {})
    columns = ctx["config"]["dataset"].get("columns", {})
    slide_col = columns.get("slide_id", "slide_id")
    sample_col = columns.get("sample_id", slide_col)
    fov_col = columns.get("fov_id", sample_col)
    group_col = cfg.get("group_column", "patient_id")
    split_col = cfg.get("split_column", "split")
    required = {
        slide_col,
        sample_col,
        fov_col,
        group_col,
        columns.get("he_path", "he_path"),
        columns.get("target_path", "target_path"),
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Slide manifest is missing columns: {', '.join(missing)}")
    for row in rows:
        sample_id, slide_name, fov_name = _canonical_identifiers(ctx, row)
        row["sample_id"] = sample_id
        row["slide_name"] = slide_name
        row["fov_name"] = fov_name
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Sample IDs must be unique in {source}")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get(group_col, "")).strip()
        if not group:
            raise ValueError(f"Empty {group_col!r} for sample {row.get(sample_col)!r}")
        groups[group].append(index)
    existing = [str(row.get(split_col, "")).strip() for row in rows]
    reuse = bool(cfg.get("reuse_existing", True)) and all(value in {"train", "valid", "test"} for value in existing)
    if reuse:
        assignments = {group: rows[indexes[0]][split_col] for group, indexes in groups.items()}
        for group, indexes in groups.items():
            if len({rows[index][split_col] for index in indexes}) != 1:
                raise ValueError(f"Group {group!r} appears in more than one existing split")
    else:
        ratios = cfg.get("ratios", {"train": 0.7, "valid": 0.1, "test": 0.2})
        values = np.asarray([float(ratios[name]) for name in ("train", "valid", "test")])
        if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("split ratios must sum to 1")
        # Retain first-occurrence order and NumPy's legacy RandomState because
        # this is the exact shuffle used by the original cohort scripts.
        ordered = np.asarray(list(groups), dtype=object)
        np.random.RandomState(int(cfg.get("seed", 42))).shuffle(ordered)
        n_groups = len(ordered)
        n_train = int(n_groups * values[0])
        n_valid = int(n_groups * values[1])
        assignments = {}
        for index, group in enumerate(ordered.tolist()):
            assignments[group] = "train" if index < n_train else "valid" if index < n_train + n_valid else "test"
    for group, indexes in groups.items():
        for index in indexes:
            rows[index][split_col] = assignments[group]
    destination = ctx["output_root"] / "manifests" / "slides_with_split.csv"
    fields = list(rows[0])
    if split_col not in fields:
        fields.append(split_col)
    _write_csv(destination, rows, fields)
    counts = {name: sum(row[split_col] == name for row in rows) for name in ("train", "valid", "test")}
    group_counts = {name: sum(value == name for value in assignments.values()) for name in counts}
    return _stage_report(
        ctx,
        "split",
        inputs=[source],
        outputs=[destination],
        details={"reused_existing_assignments": reuse, "slide_counts": counts, "group_counts": group_counts},
    )


def _register(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "manifests" / "slides_with_split.csv", "split manifest")
    rows = _read_csv(source)
    cfg = ctx["config"].get("registration", {})
    columns = ctx["config"]["dataset"].get("columns", {})
    he_col = columns.get("he_path", "he_path")
    target_col = columns.get("target_path", "target_path")
    method = str(cfg.get("method", "valis")).strip().lower().replace("-", "_")
    if method not in {"valis", "identity", "pre_registered"}:
        raise ValueError("registration.method must be one of: valis, identity, pre_registered")
    registration = None
    if method == "valis":
        try:
            from valis import registration as valis_registration
        except ImportError as exc:
            raise RuntimeError("Registration requires valis-wsi; install requirements-preprocessing.txt") from exc
        registration = valis_registration
    registered_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        he_path = _require_file(_raw_path(ctx, row[he_col]), f"H&E image for {sample_id}")
        target_path = _require_file(_raw_path(ctx, row[target_col]), f"multiplex image for {sample_id}")
        if method in {"identity", "pre_registered"}:
            registered_rows.append(
                {
                    **row,
                    "registered_he_path": str(he_path),
                    "registered_target_path": str(target_path),
                    "registration_error_csv": "",
                    "registration_status": "pre_registered",
                }
            )
            continue
        pair_dir = ctx["output_root"] / "registration" / sample_id
        input_dir = pair_dir / "inputs"
        result_dir = pair_dir / "valis"
        registered_dir = pair_dir / "registered"
        input_dir.mkdir(parents=True, exist_ok=True)
        registered_dir.mkdir(parents=True, exist_ok=True)
        he_link = input_dir / f"he{he_path.suffix}"
        target_link = input_dir / f"codex{target_path.suffix}"
        for link, original in ((he_link, he_path), (target_link, target_path)):
            if not link.exists():
                link.symlink_to(original)
        registered_he = registered_dir / "he_registered.ome.tiff"
        error_csv = result_dir / "registration_error.csv"
        if not registered_he.exists() or not bool(cfg.get("skip_existing", True)):
            assert registration is not None
            registrar = registration.Valis(
                str(input_dir),
                str(result_dir),
                reference_img_f=target_link.name,
                align_to_reference=bool(cfg.get("align_to_reference", True)),
            )
            _, _, error_df = registrar.register()
            error_csv.parent.mkdir(parents=True, exist_ok=True)
            error_df.to_csv(error_csv, index=False)
            slide = registrar.get_slide(he_link.name)
            slide.warp_and_save_slide(
                dst_f=str(registered_he),
                level=int(cfg.get("level", 0)),
                non_rigid=bool(cfg.get("non_rigid", True)),
                crop=str(cfg.get("crop", "reference")),
                Q=int(cfg.get("quality", 85)),
            )
        registered_rows.append(
            {
                **row,
                "registered_he_path": str(registered_he),
                "registered_target_path": str(target_path),
                "registration_error_csv": str(error_csv),
                "registration_status": "completed",
            }
        )
    try:
        if registration is not None:
            registration.kill_jvm()
    except Exception:
        pass
    destination = ctx["output_root"] / "manifests" / "registered_slides.csv"
    _write_csv(destination, registered_rows)
    return _stage_report(
        ctx,
        "register",
        inputs=[source],
        outputs=[destination],
        details={
            "n_registered": len(registered_rows),
            "method": method,
            "reference_modality": cfg.get("reference_modality", "multiplex"),
            "moving_modality": cfg.get("moving_modality", "H&E"),
        },
    )


def _read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".jpg", ".jpeg", ".png"}:
        from PIL import Image

        return np.asarray(Image.open(path))
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("TIFF input requires tifffile; install requirements-preprocessing.txt") from exc
    return np.asarray(tifffile.imread(path))


def _channel_last(array: np.ndarray, expected_channels: int, axis: str | int = "auto") -> np.ndarray:
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D/3D image, received shape {array.shape}")
    if axis != "auto":
        return np.moveaxis(array, int(axis), -1)
    candidates = [index for index, size in enumerate(array.shape) if size == expected_channels]
    if len(candidates) == 1:
        return np.moveaxis(array, candidates[0], -1)
    if array.shape[-1] <= expected_channels:
        return array
    if array.shape[0] <= expected_channels:
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"Cannot infer channel axis for shape {array.shape}; set tiling.target_channel_axis")


def _scale_uint8(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.uint8(np.clip((data - low) / (high - low), 0.0, 1.0) * 255)


def _registration_qc(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "manifests" / "registered_slides.csv", "registered manifest")
    rows = _read_csv(source)
    destination = ctx["output_root"] / "registration_qc" / "registration_qc.csv"
    cfg = ctx["config"].get("registration_qc", {})
    previous = {row["sample_id"]: row for row in _read_csv(destination)} if destination.exists() else {}
    if not bool(cfg.get("enabled", True)):
        output_rows = [
            {**row, "overlay_path": "", "decision": "accepted", "notes": "QC disabled by config"}
            for row in rows
        ]
        _write_csv(destination, output_rows)
        return _stage_report(
            ctx,
            "registration-qc",
            inputs=[source],
            outputs=[destination],
            details={"enabled": False, "decision_counts": {"accepted": len(output_rows)}},
        )
    raw_count = int(ctx["config"].get("tiling", {}).get("raw_channel_count", 92))
    raw_nuclear = int(cfg.get("raw_nuclear_channel_index", 91))
    max_side = int(cfg.get("thumbnail_max_side", 1024))
    output_rows: list[dict[str, Any]] = []
    from PIL import Image

    for row in rows:
        sample_id = row["sample_id"]
        he = _read_array(Path(row["registered_he_path"]))
        if he.ndim == 2:
            he = np.repeat(he[..., None], 3, axis=-1)
        elif he.ndim == 3 and he.shape[0] in {3, 4} and he.shape[-1] not in {3, 4}:
            he = np.moveaxis(he, 0, -1)
        target = _channel_last(
            _read_array(Path(row["registered_target_path"])),
            raw_count,
            ctx["config"].get("tiling", {}).get("target_channel_axis", "auto"),
        )
        if raw_nuclear >= target.shape[-1]:
            raise ValueError(f"raw_nuclear_channel_index={raw_nuclear} exceeds target channels for {sample_id}")
        h_ref = 255 - _scale_uint8(he[..., 0])
        nuclear = _scale_uint8(target[..., raw_nuclear])
        height = min(h_ref.shape[0], nuclear.shape[0])
        width = min(h_ref.shape[1], nuclear.shape[1])
        h_ref, nuclear = h_ref[:height, :width], nuclear[:height, :width]
        scale = min(1.0, max_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        h_thumb = np.asarray(Image.fromarray(h_ref).resize(size, Image.Resampling.BILINEAR))
        n_thumb = np.asarray(Image.fromarray(nuclear).resize(size, Image.Resampling.BILINEAR))
        overlay = np.stack([h_thumb, n_thumb, np.zeros_like(h_thumb)], axis=-1)
        overlay_path = ctx["output_root"] / "registration_qc" / "overlays" / f"{sample_id}.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(overlay_path, quality=90)
        old = previous.get(sample_id, {})
        output_rows.append(
            {
                **row,
                "overlay_path": str(overlay_path),
                "decision": old.get(
                    "decision", "pending" if bool(cfg.get("require_manual_acceptance", True)) else "accepted"
                ),
                "notes": old.get("notes", ""),
            }
        )
    _write_csv(destination, output_rows)
    decisions = defaultdict(int)
    for row in output_rows:
        decisions[str(row["decision"]).strip().lower()] += 1
    return _stage_report(
        ctx,
        "registration-qc",
        inputs=[source],
        outputs=[destination],
        details={
            "enabled": True,
            "decision_counts": dict(decisions),
            "instruction": "Inspect each overlay and set decision to accepted or rejected before tiling.",
        },
    )


def _accepted_registered_rows(ctx: dict[str, Any]) -> tuple[list[dict[str, str]], Path]:
    source = _require_file(ctx["output_root"] / "registration_qc" / "registration_qc.csv", "registration QC table")
    rows = _read_csv(source)
    for row in rows:
        if not all(row.get(key) for key in ("sample_id", "slide_name", "fov_name")):
            sample_id, slide_name, fov_name = _canonical_identifiers(ctx, row)
            row["sample_id"] = sample_id
            row["slide_name"] = slide_name
            row["fov_name"] = fov_name
    cfg = ctx["config"].get("registration_qc", {})
    if bool(cfg.get("require_manual_acceptance", True)):
        pending = [row["sample_id"] for row in rows if row.get("decision", "").strip().lower() == "pending"]
        invalid = [
            row["sample_id"]
            for row in rows
            if row.get("decision", "").strip().lower() not in {"accepted", "rejected", "pending"}
        ]
        if pending or invalid:
            raise RuntimeError(
                f"Registration QC is incomplete: {len(pending)} pending and {len(invalid)} invalid decisions in {source}"
            )
        rows = [row for row in rows if row.get("decision", "").strip().lower() == "accepted"]
    return rows, source


def _tile(ctx: dict[str, Any]) -> dict[str, Any]:
    rows, source = _accepted_registered_rows(ctx)
    cfg = ctx["config"].get("tiling", {})
    patch_size = int(cfg.get("patch_size", 256))
    stride = int(cfg.get("stride", patch_size))
    if stride != patch_size:
        raise ValueError("The FOV-level cell pipeline requires non-overlapping patches")
    raw_count = int(cfg.get("raw_channel_count", 92))
    fov_size_value = cfg.get("fov_size_px")
    fov_size = int(fov_size_value) if fov_size_value is not None else None
    if fov_size is not None and (fov_size < patch_size or fov_size % patch_size):
        raise ValueError("tiling.fov_size_px must be null or a positive multiple of patch_size")
    channels = _channels(ctx)
    retained_indices = [int(value) for value in cfg.get("retained_channel_indices", range(raw_count))]
    if len(retained_indices) != len(channels):
        raise ValueError("tiling.retained_channel_indices must match channel_names_file length")
    if not retained_indices or min(retained_indices) < 0:
        raise ValueError("tiling.retained_channel_indices must contain non-negative channel indices")
    columns = ctx["config"]["dataset"].get("columns", {})
    origin_y_col = str(columns.get("origin_y_px", "origin_y_px"))
    origin_x_col = str(columns.get("origin_x_px", "origin_x_px"))
    from PIL import Image

    patch_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        slide_name = row["slide_name"]
        fov_name = row["fov_name"]
        origin_y = _as_nonnegative_int(row.get(origin_y_col, 0), origin_y_col)
        origin_x = _as_nonnegative_int(row.get(origin_x_col, 0), origin_x_col)
        if origin_y % patch_size or origin_x % patch_size:
            raise ValueError(f"FOV origins for {sample_id} must be multiples of patch_size={patch_size}")
        he = _read_array(Path(row["registered_he_path"]))
        if he.ndim == 2:
            he = np.repeat(he[..., None], 3, axis=-1)
        elif he.ndim == 3 and he.shape[0] in {3, 4} and he.shape[-1] not in {3, 4}:
            he = np.moveaxis(he, 0, -1)
        he = he[..., :3]
        if he.dtype != np.uint8:
            he = _scale_uint8(he)
        target = _channel_last(
            _read_array(Path(row["registered_target_path"])), raw_count, cfg.get("target_channel_axis", "auto")
        )
        if max(retained_indices) >= target.shape[-1]:
            raise ValueError(f"Retained channel index exceeds available channels for {sample_id}: {target.shape}")
        target = target[..., retained_indices]
        height = min(he.shape[0], target.shape[0])
        width = min(he.shape[1], target.shape[1])
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                if fov_size is None:
                    tile_fov_name = fov_name
                    fov_row, fov_col = y // patch_size, x // patch_size
                else:
                    fov_grid_row, fov_grid_col = y // fov_size, x // fov_size
                    tile_fov_name = f"{fov_name}_fov_{fov_grid_row:03d}_{fov_grid_col:03d}"
                    fov_row = (y % fov_size) // patch_size
                    fov_col = (x % fov_size) // patch_size
                rr, cc = origin_y // patch_size + fov_row, origin_x // patch_size + fov_col
                if fov_size is not None:
                    rr += (y // fov_size) * (fov_size // patch_size)
                    cc += (x // fov_size) * (fov_size // patch_size)
                stem = f"{sample_id}_patch_{fov_row:03d}_{fov_col:03d}"
                if fov_size is not None:
                    stem = f"{sample_id}_fov_{fov_grid_row:03d}_{fov_grid_col:03d}_patch_{fov_row:03d}_{fov_col:03d}"
                he_path = ctx["output_root"] / "patches" / "he" / tile_fov_name / f"{stem}.jpg"
                raw_path = ctx["output_root"] / "patches" / "target_raw" / tile_fov_name / f"{stem}.npy"
                he_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(he[y : y + patch_size, x : x + patch_size]).save(he_path, quality=95)
                np.save(raw_path, target[y : y + patch_size, x : x + patch_size])
                patch_rows.append(
                    {
                        "slide_name": slide_name,
                        "sample_id": sample_id,
                        "fov_name": tile_fov_name,
                        "group_id": row.get(ctx["config"].get("split", {}).get("group_column", "group_id"), ""),
                        "row": rr,
                        "col": cc,
                        "fov_row": fov_row,
                        "fov_col": fov_col,
                        "split": row["split"],
                        "image_path": _relative(he_path, ctx["output_root"]),
                        "target_raw_path": _relative(raw_path, ctx["output_root"]),
                    }
                )
    keys = [(row["slide_name"], row["row"], row["col"]) for row in patch_rows]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate (slide_name,row,col) coordinates were generated. Give each FOV a unique slide_id "
            "or provide non-overlapping origin_y_px/origin_x_px values in the sample manifest."
        )
    destination = ctx["output_root"] / "manifests" / "patches_raw.csv"
    channel_destination = ctx["output_root"] / "channel_names.json"
    _write_csv(destination, patch_rows)
    _write_json(channel_destination, channels)
    return _stage_report(
        ctx,
        "tile",
        inputs=[source, ctx["channels_file"]],
        outputs=[destination, channel_destination],
        details={
            "n_input_samples": len(rows),
            "n_fovs": len({row["fov_name"] for row in patch_rows}),
            "n_patches": len(patch_rows),
            "patch_size": patch_size,
            "stride": stride,
            "fov_size_px": fov_size,
        },
    )


def _legacy_normalize(
    array: np.ndarray,
    quantiles: np.ndarray,
    *,
    divide_by_log2: bool = False,
    output_dtype: str = "uint8",
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    q = np.asarray(quantiles, dtype=np.float32)
    if values.shape[-1] != q.size:
        raise ValueError("Target channel count does not match quantile count")
    safe = np.where(q > 0, q, 1.0)
    normalized = np.log1p(np.clip(values, 0, safe) / safe)
    if divide_by_log2:
        normalized /= np.log(2.0)
    scaled = np.clip(normalized * 255.0, 0.0, 255.0)
    dtype = array.dtype if output_dtype == "preserve_input" else np.dtype(output_dtype)
    return scaled.astype(dtype)


def _normalize(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "manifests" / "patches_raw.csv", "raw patch manifest")
    rows = _read_csv(source)
    channels = _channels(ctx)
    cfg = ctx["config"].get("normalization", {})
    transform = str(cfg.get("transform", "legacy_log1p")).strip().lower()
    if transform in {"identity", "none", "pre_normalized"}:
        output_rows = [{**row, "target_path": row["target_raw_path"]} for row in rows]
        stats_path = ctx["output_root"] / "channel_stats.json"
        _write_json(
            stats_path,
            {
                "schema": "heprobench_channel_normalization_v1",
                "transform": "identity",
                "reason": cfg.get("reason", "Input values are already in the benchmark target range."),
                "channels": {name: {} for name in channels},
            },
        )
        destination = ctx["output_root"] / "manifests" / "patches_normalized.csv"
        _write_csv(destination, output_rows)
        return _stage_report(
            ctx,
            "normalize",
            inputs=[source],
            outputs=[stats_path, destination],
            details={"n_normalized_patches": len(output_rows), "transform": "identity"},
        )
    if transform != "legacy_log1p":
        raise ValueError("normalization.transform must be legacy_log1p or identity")
    percentile = float(cfg.get("percentile", 99.9))
    samples_per_patch = int(cfg.get("samples_per_patch", 10000))
    fit_rows = [row for row in rows if row["split"] == str(cfg.get("fit_split", "train"))]
    if not fit_rows:
        raise ValueError("No training patches are available for normalization fitting")
    try:
        from tdigest import TDigest
    except ImportError as exc:
        raise RuntimeError("Normalization requires tdigest; install requirements-preprocessing.txt") from exc
    digests = [TDigest() for _ in channels]
    for row in fit_rows:
        array = np.load(ctx["output_root"] / row["target_raw_path"], mmap_mode="r")
        flat = np.asarray(array).reshape(-1, len(channels))
        stride = max(1, flat.shape[0] // samples_per_patch)
        sampled = flat[::stride][:samples_per_patch]
        for index, digest in enumerate(digests):
            digest.batch_update(sampled[:, index].astype(float).tolist())
    quantiles = np.asarray([digest.percentile(percentile) for digest in digests], dtype=np.float32)
    divide_by_log2 = bool(cfg.get("divide_by_log2", False))
    output_dtype = str(cfg.get("output_dtype", "uint8"))
    if output_dtype != "preserve_input":
        np.dtype(output_dtype)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_path = ctx["output_root"] / row["target_raw_path"]
        norm_path = (
            ctx["output_root"]
            / "patches"
            / "target_norm"
            / row.get("fov_name", row["slide_name"])
            / Path(row["target_raw_path"]).name
        )
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.load(raw_path)
        np.save(
            norm_path,
            _legacy_normalize(
                raw,
                quantiles,
                divide_by_log2=divide_by_log2,
                output_dtype=output_dtype,
            ),
        )
        output_rows.append({**row, "target_path": _relative(norm_path, ctx["output_root"])})
    stats_path = ctx["output_root"] / "channel_stats.json"
    stats = {
        "schema": "heprobench_channel_normalization_v1",
        "fit_split": cfg.get("fit_split", "train"),
        "percentile": percentile,
        "estimator": "TDigest over deterministic per-patch stride samples",
        "samples_per_patch": samples_per_patch,
        "transform": "log1p(clip(x, 0, q) / q) * 255",
        "divide_by_log2": divide_by_log2,
        "output_dtype": output_dtype,
        "channels": {name: {"q": float(value)} for name, value in zip(channels, quantiles)},
    }
    _write_json(stats_path, stats)
    destination = ctx["output_root"] / "manifests" / "patches_normalized.csv"
    _write_csv(destination, output_rows)
    return _stage_report(
        ctx,
        "normalize",
        inputs=[source],
        outputs=[stats_path, destination],
        details={
            "n_fit_patches": len(fit_rows),
            "n_normalized_patches": len(output_rows),
            "divide_by_log2": divide_by_log2,
            "output_dtype": output_dtype,
        },
    )


def _nmi(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    histogram, _, _ = np.histogram2d(x.reshape(-1), y.reshape(-1), bins=bins, range=((0, 256), (0, 256)))
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    joint = histogram / total
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    hxy = -float(np.sum(joint[joint > 0] * np.log(joint[joint > 0])))
    return float(2 * (hx + hy - hxy) / (hx + hy)) if hx + hy > 0 else 0.0


def _robust_nmi(x: np.ndarray, y: np.ndarray, *, sigma: float = 1.5, bins: int = 16, min_foreground: int = 100) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Patch QC requires opencv-python-headless; install requirements-preprocessing.txt") from exc
    x_blur = cv2.GaussianBlur(np.asarray(x, dtype=np.uint8), (0, 0), sigmaX=sigma)
    y_blur = cv2.GaussianBlur(np.asarray(y, dtype=np.uint8), (0, 0), sigmaX=sigma)
    step = max(1, 256 // bins)
    x_quant, y_quant = x_blur // step, y_blur // step
    foreground = (x_quant > 0) | (y_quant > 0)
    if int(foreground.sum()) < min_foreground:
        return 0.0
    return _nmi(x_quant[foreground] * step, y_quant[foreground] * step, bins)


def _patch_qc(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "manifests" / "patches_normalized.csv", "normalized patch manifest")
    rows = _read_csv(source)
    channels = _channels(ctx)
    cfg = ctx["config"].get("patch_qc", {})
    if not bool(cfg.get("enabled", True)):
        output_rows = [
            {
                **row,
                "nuclear_std": "",
                "nmi": "",
                "robust_nmi": "",
                "pass_nuclear_std": 1,
                "pass_robust_nmi": 1,
                "qc_pass": 1,
            }
            for row in rows
        ]
        qc_path = ctx["output_root"] / "manifests" / "patch_qc.csv"
        _write_csv(qc_path, output_rows)
        split_outputs: list[Path] = []
        for split in ("train", "valid", "test"):
            destination = ctx["output_root"] / "splits" / f"{split}.csv"
            selected = [row for row in output_rows if row["split"] == split]
            _write_csv(destination, selected, output_rows[0].keys() if output_rows else [])
            split_outputs.append(destination)
        return _stage_report(
            ctx,
            "patch-qc",
            inputs=[source],
            outputs=[qc_path, *split_outputs],
            details={"enabled": False, "n_total": len(rows), "n_pass": len(rows)},
        )
    nuclear_index = channels.index(str(cfg.get("nuclear_channel", "DRAQ5")))
    std_threshold = float(cfg.get("nuclear_std_min", 11.0))
    nmi_threshold = float(cfg.get("robust_nmi_min", 0.03))
    sigma = float(cfg.get("gaussian_sigma", 1.5))
    bins = int(cfg.get("robust_nmi_bins", 16))
    min_foreground = int(cfg.get("min_foreground_pixels", 100))
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        from PIL import Image

        he = np.asarray(Image.open(ctx["output_root"] / row["image_path"]).convert("RGB"))
        target = np.load(ctx["output_root"] / row["target_path"])
        he_reference = 255 - he[..., int(cfg.get("he_inverse_channel", 0))]
        nuclear = target[..., nuclear_index]
        nuclear_std = float(np.std(nuclear))
        standard_nmi = _nmi(he_reference, nuclear, int(cfg.get("diagnostic_nmi_bins", 64)))
        robust = _robust_nmi(he_reference, nuclear, sigma=sigma, bins=bins, min_foreground=min_foreground)
        pass_std, pass_nmi = nuclear_std > std_threshold, robust > nmi_threshold
        output_rows.append(
            {
                **row,
                "nuclear_std": nuclear_std,
                "nmi": standard_nmi,
                "robust_nmi": robust,
                "pass_nuclear_std": int(pass_std),
                "pass_robust_nmi": int(pass_nmi),
                "qc_pass": int(pass_std and pass_nmi),
            }
        )
    qc_path = ctx["output_root"] / "manifests" / "patch_qc.csv"
    _write_csv(qc_path, output_rows)
    split_outputs: list[Path] = []
    for split in ("train", "valid", "test"):
        destination = ctx["output_root"] / "splits" / f"{split}.csv"
        selected = [row for row in output_rows if row["split"] == split and int(row["qc_pass"]) == 1]
        _write_csv(destination, selected, output_rows[0].keys() if output_rows else [])
        split_outputs.append(destination)
    return _stage_report(
        ctx,
        "patch-qc",
        inputs=[source],
        outputs=[qc_path, *split_outputs],
        details={
            "enabled": True,
            "n_total": len(rows),
            "n_pass": sum(int(row["qc_pass"]) for row in output_rows),
            "nuclear_channel": channels[nuclear_index],
            "nuclear_std_min_strict": std_threshold,
            "robust_nmi_min_strict": nmi_threshold,
        },
    )


def _minmax(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / (high - low + 1e-6)


def _segment(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "manifests" / "patches_normalized.csv", "normalized patch manifest")
    rows = _read_csv(source)
    channels = _channels(ctx)
    cfg = ctx["config"].get("segmentation", {})
    nuclear_index = channels.index(str(cfg.get("nuclear_channel", "DRAQ5")))
    boundary_indices = [channels.index(name) for name in cfg.get("boundary_channels", ["Na-K-ATPase"])]
    patch_size = int(ctx["config"].get("tiling", {}).get("patch_size", 256))
    try:
        from deepcell.applications import Mesmer
    except ImportError as exc:
        raise RuntimeError("Cell segmentation requires DeepCell/Mesmer; see requirements-preprocessing.txt") from exc
    app = Mesmer()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("fov_name", row["slide_name"])].append(row)
    output_rows: list[dict[str, Any]] = []
    next_cell_id: dict[str, int] = defaultdict(int)
    for fov_name, fov_rows in sorted(grouped.items()):
        slide_names = {row["slide_name"] for row in fov_rows}
        if len(slide_names) != 1:
            raise ValueError(f"FOV {fov_name!r} is associated with multiple slide names")
        slide_name = next(iter(slide_names))
        height = (max(int(row.get("fov_row", row["row"])) for row in fov_rows) + 1) * patch_size
        width = (max(int(row.get("fov_col", row["col"])) for row in fov_rows) + 1) * patch_size
        nuclear = np.zeros((height, width), dtype=np.float32)
        boundary_channels = np.zeros((height, width, len(boundary_indices)), dtype=np.float32)
        for row in fov_rows:
            target = np.load(ctx["output_root"] / row["target_path"])
            y = int(row.get("fov_row", row["row"])) * patch_size
            x = int(row.get("fov_col", row["col"])) * patch_size
            nuclear[y : y + patch_size, x : x + patch_size] = target[..., nuclear_index]
            boundary_channels[y : y + patch_size, x : x + patch_size, :] = target[..., boundary_indices]
        boundary = np.max(
            np.stack([_minmax(boundary_channels[..., index]) for index in range(len(boundary_indices))]),
            axis=0,
        )
        model_input = np.stack([_minmax(nuclear), _minmax(boundary)], axis=-1)[None]
        predict_kwargs: dict[str, Any] = {
            "postprocess_kwargs_whole_cell": {
                "maxima_algorithm": str(cfg.get("maxima_algorithm", "peak_local_max"))
            }
        }
        if cfg.get("image_mpp", 0.69) is not None:
            predict_kwargs["image_mpp"] = float(cfg["image_mpp"])
        prediction = app.predict(model_input, **predict_kwargs)
        labels = np.asarray(prediction).squeeze().astype(np.int32)
        positive = labels > 0
        labels[positive] += next_cell_id[slide_name]
        next_cell_id[slide_name] = int(labels.max(initial=next_cell_id[slide_name]))
        for row in fov_rows:
            y = int(row.get("fov_row", row["row"])) * patch_size
            x = int(row.get("fov_col", row["col"])) * patch_size
            mask_path = ctx["output_root"] / "patches" / "masks" / fov_name / Path(row["target_path"]).name
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_path, labels[y : y + patch_size, x : x + patch_size])
            output_rows.append({**row, "mask_path": _relative(mask_path, ctx["output_root"])})
    destination = ctx["output_root"] / "manifests" / "patches_with_masks.csv"
    _write_csv(destination, output_rows)
    qc_rows = _read_csv(_require_file(ctx["output_root"] / "manifests" / "patch_qc.csv", "patch QC table"))
    mask_lookup = {(row["slide_name"], row["row"], row["col"]): row["mask_path"] for row in output_rows}
    split_outputs: list[Path] = []
    for split in ("train", "valid", "test"):
        selected = []
        for row in qc_rows:
            if row["split"] != split or int(row["qc_pass"]) != 1:
                continue
            selected.append({**row, "mask_path": mask_lookup[(row["slide_name"], row["row"], row["col"])]})
        split_path = ctx["output_root"] / "splits" / f"{split}.csv"
        _write_csv(split_path, selected, selected[0].keys() if selected else list(qc_rows[0]) + ["mask_path"])
        split_outputs.append(split_path)
    return _stage_report(
        ctx,
        "segment",
        inputs=[source],
        outputs=[destination, *split_outputs],
        details={
            "n_fovs": len(grouped),
            "model": "DeepCell Mesmer",
            "nuclear_channel": channels[nuclear_index],
            "boundary_channels": [channels[index] for index in boundary_indices],
            "image_mpp": cfg.get("image_mpp", 0.69),
            "cell_ids_unique_within": "slide_name",
        },
    )


def _cell_extract(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "manifests" / "patches_with_masks.csv", "masked patch manifest")
    rows = _read_csv(source)
    qc_rows = _read_csv(_require_file(ctx["output_root"] / "manifests" / "patch_qc.csv", "patch QC table"))
    channels = _channels(ctx)
    patch_size = int(ctx["config"].get("tiling", {}).get("patch_size", 256))
    passing = {
        (
            row.get("fov_name", row["slide_name"]),
            int(row.get("fov_row", row["row"])),
            int(row.get("fov_col", row["col"])),
        ): row["split"]
        for row in qc_rows
        if int(row["qc_pass"]) == 1
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("fov_name", row["slide_name"])].append(row)
    cells: list[dict[str, Any]] = []
    for fov_name, fov_rows in grouped.items():
        slide_names = {row["slide_name"] for row in fov_rows}
        if len(slide_names) != 1:
            raise ValueError(f"FOV {fov_name!r} is associated with multiple slide names")
        slide_name = next(iter(slide_names))
        origins_y = {
            (int(row["row"]) - int(row.get("fov_row", row["row"]))) * patch_size for row in fov_rows
        }
        origins_x = {
            (int(row["col"]) - int(row.get("fov_col", row["col"]))) * patch_size for row in fov_rows
        }
        if len(origins_y) != 1 or len(origins_x) != 1:
            raise ValueError(f"Inconsistent global origin across patches in FOV {fov_name!r}")
        origin_y, origin_x = next(iter(origins_y)), next(iter(origins_x))
        id_parts = []
        for row in fov_rows:
            ids = np.unique(np.load(ctx["output_root"] / row["mask_path"]))
            id_parts.append(ids[ids > 0].astype(np.int64, copy=False))
        all_cell_ids = np.unique(np.concatenate(id_parts)) if id_parts else np.asarray([], dtype=np.int64)
        counts = np.zeros(len(all_cell_ids), dtype=np.int64)
        sum_y = np.zeros(len(all_cell_ids), dtype=np.float64)
        sum_x = np.zeros(len(all_cell_ids), dtype=np.float64)
        sums = np.zeros((len(all_cell_ids), len(channels)), dtype=np.float64)
        for row in fov_rows:
            mask = np.load(ctx["output_root"] / row["mask_path"]).astype(np.int64)
            target = np.load(ctx["output_root"] / row["target_path"])
            ids = mask.reshape(-1)
            foreground = ids > 0
            positions = np.searchsorted(all_cell_ids, ids[foreground])
            counts += np.bincount(positions, minlength=len(all_cell_ids))
            yy, xx = np.indices(mask.shape)
            y_offset = int(row.get("fov_row", row["row"])) * patch_size
            x_offset = int(row.get("fov_col", row["col"])) * patch_size
            sum_y += np.bincount(
                positions,
                weights=(yy + y_offset).reshape(-1)[foreground],
                minlength=len(all_cell_ids),
            )
            sum_x += np.bincount(
                positions,
                weights=(xx + x_offset).reshape(-1)[foreground],
                minlength=len(all_cell_ids),
            )
            for index in range(len(channels)):
                sums[:, index] += np.bincount(
                    positions,
                    weights=target[..., index].reshape(-1)[foreground],
                    minlength=len(all_cell_ids),
                )
        for cell_index, cell_id in enumerate(all_cell_ids.tolist()):
            centroid_y = sum_y[cell_index] / counts[cell_index]
            centroid_x = sum_x[cell_index] / counts[cell_index]
            rr, cc = int(centroid_y // patch_size), int(centroid_x // patch_size)
            split = passing.get((fov_name, rr, cc))
            if split is None:
                continue
            cell: dict[str, Any] = {
                "slide_name": slide_name,
                "fov_name": fov_name,
                "global_cell_id": int(cell_id),
                "split": split,
                "area": int(counts[cell_index]),
                "centroid_y": float(centroid_y + origin_y),
                "centroid_x": float(centroid_x + origin_x),
            }
            for index, channel in enumerate(channels):
                cell[channel] = float(sums[cell_index, index] / counts[cell_index])
            cells.append(cell)
    destination = ctx["output_root"] / "cell_annotations_continuous.csv"
    fields = ["slide_name", "fov_name", "global_cell_id", "split", "area", "centroid_y", "centroid_x", *channels]
    _write_csv(destination, cells, fields)
    return _stage_report(
        ctx,
        "cell-extract",
        inputs=[source],
        outputs=[destination],
        details={"n_cells_in_qc_patches": len(cells), "assignment": "cell centroid selects one QC-passing patch"},
    )


def _gate(ctx: dict[str, Any]) -> dict[str, Any]:
    source = _require_file(ctx["output_root"] / "cell_annotations_continuous.csv", "continuous cell table")
    rows = _read_csv(source)
    channels = _channels(ctx)
    cfg = ctx["config"].get("gating", {})
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError("GMM gating requires scikit-learn") from exc
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    posterior_threshold = float(cfg.get("posterior_threshold", 0.5))
    positive_only = bool(cfg.get("positive_only_fit", True))
    max_cells = int(cfg.get("max_cells_per_marker", 300000))
    min_cells = int(cfg.get("min_cells_to_fit", 2000))
    separation_min = float(cfg.get("separation_min", 1.5))
    rate_min = float(cfg.get("positive_rate_min", 0.01))
    rate_max = float(cfg.get("positive_rate_max", 0.99))
    params: dict[str, Any] = {}
    qc_rows: list[dict[str, Any]] = []
    for channel in channels:
        all_values = np.asarray([float(row[channel]) for row in rows], dtype=np.float64)
        fit_values = all_values[np.isfinite(all_values)]
        if positive_only:
            fit_values = fit_values[fit_values > 0]
        if fit_values.size > max_cells:
            fit_values = rng.choice(fit_values, size=max_cells, replace=False)
        if fit_values.size < min_cells:
            for row in rows:
                row[f"{channel}_pos"] = ""
            params[channel] = {"status": "insufficient_cells", "n_samples_fit": int(fit_values.size)}
            qc_rows.append(
                {"marker": channel, "n_samples_fit": int(fit_values.size), "separation": "", "positive_rate": "", "gating_valid": 0}
            )
            continue
        model = GaussianMixture(n_components=2, random_state=seed).fit(fit_values.reshape(-1, 1))
        means = model.means_.reshape(-1)
        stds = np.sqrt(model.covariances_.reshape(-1))
        weights = model.weights_.reshape(-1)
        high = int(np.argmax(means))
        finite = np.isfinite(all_values)
        probabilities = np.full(all_values.shape, np.nan, dtype=np.float64)
        probabilities[finite] = model.predict_proba(all_values[finite].reshape(-1, 1))[:, high]
        labels = probabilities > posterior_threshold
        for row, label, is_finite in zip(rows, labels.tolist(), finite.tolist()):
            row[f"{channel}_pos"] = int(label) if is_finite else ""
        low = 1 - high
        separation = float(abs(means[high] - means[low]) / np.sqrt(0.5 * (stds[high] ** 2 + stds[low] ** 2)))
        positive_rate = float(np.mean(labels[finite]))
        valid = separation >= separation_min and rate_min <= positive_rate <= rate_max
        params[channel] = {
            "status": "ok",
            "n_samples_fit": int(fit_values.size),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "weights": weights.tolist(),
            "high_component": high,
            "positive_only_fit": positive_only,
            "posterior_threshold": posterior_threshold,
        }
        qc_rows.append(
            {
                "marker": channel,
                "n_samples_fit": int(fit_values.size),
                "separation": separation,
                "positive_rate": positive_rate,
                "gating_valid": int(valid),
            }
        )
    destination = ctx["output_root"] / "cell_annotations.csv"
    fields = ["slide_name", "fov_name", "global_cell_id", "split", "area", "centroid_y", "centroid_x"]
    for channel in channels:
        fields.extend([channel, f"{channel}_pos"])
    _write_csv(destination, rows, fields)
    params_path = ctx["output_root"] / "gmm_params.json"
    qc_path = ctx["output_root"] / "gmm_marker_qc.csv"
    _write_json(params_path, params)
    _write_csv(qc_path, qc_rows)
    return _stage_report(
        ctx,
        "gate",
        inputs=[source],
        outputs=[destination, params_path, qc_path],
        details={
            "n_cells": len(rows),
            "n_gating_valid_markers": sum(int(row["gating_valid"]) for row in qc_rows),
            "separation_min": separation_min,
            "positive_rate_range": [rate_min, rate_max],
        },
    )


_RUNNERS = {
    "split": _split,
    "register": _register,
    "registration-qc": _registration_qc,
    "tile": _tile,
    "normalize": _normalize,
    "patch-qc": _patch_qc,
    "segment": _segment,
    "cell-extract": _cell_extract,
    "gate": _gate,
}


def run_preprocessing(
    config_path: str | Path,
    *,
    stages: list[str] | None = None,
    output_dir: str | Path | None = None,
    overrides: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ctx = load_preprocessing_config(config_path, output_dir=output_dir, overrides=overrides)
    warnings = _validate_preprocessing_config(ctx)
    requested = stages or ["all"]
    if "all" in requested:
        requested = list(STAGES)
    unknown = sorted(set(requested).difference(STAGES))
    if unknown:
        raise ValueError(f"Unknown preprocessing stages: {', '.join(unknown)}")
    plan = {
        "schema": SCHEMA_VERSION,
        "dataset": ctx["config"]["dataset"]["name"],
        "config": str(ctx["source"]),
        "data_root": str(ctx["data_root"]),
        "output_root": str(ctx["output_root"]),
        "stages": requested,
        "dry_run": bool(dry_run),
        "channel_count": len(_channels(ctx)),
        "access_status": ctx["config"]["dataset"].get("access", {}).get("status", "unspecified"),
        "reproduction_status": ctx["config"].get("reproduction", {}).get("status", "unspecified"),
        "warnings": warnings,
    }
    if dry_run:
        return plan
    ctx["output_root"].mkdir(parents=True, exist_ok=True)
    resolved_path = ctx["output_root"] / "resolved_preprocessing_config.json"
    _write_json(resolved_path, _resolved_config_payload(ctx))
    results = []
    for stage in requested:
        results.append(_RUNNERS[stage](ctx))
    reports = []
    for stage in STAGES:
        report_path = ctx["reports_dir"] / f"{stage}.json"
        if report_path.is_file():
            with report_path.open(encoding="utf-8") as handle:
                reports.append(json.load(handle))
    manifest = {
        **plan,
        "dry_run": False,
        "completed_at": _utc_now(),
        "stages_completed_this_run": requested,
        "reports": reports,
    }
    _write_json(ctx["output_root"] / "preprocessing_manifest.json", manifest)
    return manifest
