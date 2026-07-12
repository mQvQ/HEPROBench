from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .schema import PatchRecord


SCHEMA_VERSION = "heprobench_h5_v1"


def _string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def write_slide_h5(
    output_path: str | Path,
    predictions: np.ndarray,
    records: list[PatchRecord],
    channel_names: list[str],
    attrs: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions.ndim != 4:
        raise ValueError("predictions must have shape [N,H,W,C]")
    if predictions.shape[0] != len(records):
        raise ValueError("prediction count does not match records")
    if predictions.shape[-1] != len(channel_names):
        raise ValueError("prediction channel count does not match channel_names")
    if predictions.dtype != np.uint8:
        predictions = np.clip(predictions, 0, 255).astype(np.uint8)

    rows = np.asarray([r.row for r in records], dtype=np.int32)
    cols = np.asarray([r.col for r in records], dtype=np.int32)
    n_rows = int(rows.max()) + 1 if len(rows) else 0
    n_cols = int(cols.max()) + 1 if len(cols) else 0
    grid_index = np.full((n_rows, n_cols), -1, dtype=np.int64)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        grid_index[int(row), int(col)] = idx

    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        for key, value in attrs.items():
            if value is not None:
                h5.attrs[key] = value
        data = h5.create_group("data")
        data.create_dataset("pred", data=predictions, compression="gzip", compression_opts=4)
        data.create_dataset("rows", data=rows)
        data.create_dataset("cols", data=cols)
        data.create_dataset("grid_index", data=grid_index)
        data.create_dataset("channel_names", data=_string_array(channel_names))
        data.create_dataset("image_paths", data=_string_array([r.image_path_rel for r in records]))
        data.create_dataset("target_paths", data=_string_array([r.target_path_rel for r in records]))
        data.create_dataset("has_gt", data=np.asarray([r.target_path is not None for r in records], dtype=bool))


def read_slide_h5(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path} has unsupported schema_version={h5.attrs.get('schema_version')!r}")
        data = h5["data"]
        result = {
            "attrs": dict(h5.attrs),
            "pred": data["pred"][:],
            "rows": data["rows"][:],
            "cols": data["cols"][:],
            "grid_index": data["grid_index"][:],
            "channel_names": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in data["channel_names"][:]],
            "image_paths": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in data["image_paths"][:]],
            "target_paths": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in data["target_paths"][:]],
            "has_gt": data["has_gt"][:].astype(bool),
        }
    return result

