from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PatchRecord:
    slide_name: str
    row: int
    col: int
    image_path: Path
    target_path: Path | None
    image_path_rel: str
    target_path_rel: str
    split: str


def load_channel_names(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        values = json.load(f)
    if not isinstance(values, list) or not values or not all(isinstance(x, str) for x in values):
        raise ValueError("channel_names.json must be a non-empty JSON list of strings")
    if len(set(values)) != len(values):
        raise ValueError("channel_names.json contains duplicated channel names")
    return values


def load_records(dataset_cfg: dict) -> list[PatchRecord]:
    root = Path(dataset_cfg["root"])
    metadata_csv = Path(dataset_cfg["metadata_csv"])
    image_col = dataset_cfg.get("image_column", "image_path")
    target_col = dataset_cfg.get("target_column", "target_path")
    slide_col = dataset_cfg.get("slide_column", "slide_name")
    row_col = dataset_cfg.get("row_column", "row")
    col_col = dataset_cfg.get("col_column", "col")
    split_col = dataset_cfg.get("split_column", "split")

    records: list[PatchRecord] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {metadata_csv}")
        required = {image_col, target_col, slide_col, row_col, col_col}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"metadata.csv missing required columns: {', '.join(missing)}")
        for idx, row in enumerate(reader, start=2):
            slide_name = (row.get(slide_col) or "").strip()
            image_rel = (row.get(image_col) or "").strip()
            target_rel = (row.get(target_col) or "").strip()
            if not slide_name or not image_rel:
                raise ValueError(f"metadata.csv row {idx} has empty slide_name or image_path")
            try:
                rr = int(row.get(row_col, ""))
                cc = int(row.get(col_col, ""))
            except ValueError as exc:
                raise ValueError(f"metadata.csv row {idx} has invalid row/col") from exc
            split = (row.get(split_col) or "demo").strip()
            image_path = Path(image_rel)
            if not image_path.is_absolute():
                image_path = root / image_path
            target_path = None
            if target_rel:
                target_path = Path(target_rel)
                if not target_path.is_absolute():
                    target_path = root / target_path
            records.append(
                PatchRecord(
                    slide_name=slide_name,
                    row=rr,
                    col=cc,
                    image_path=image_path.resolve(),
                    target_path=target_path.resolve() if target_path else None,
                    image_path_rel=image_rel,
                    target_path_rel=target_rel,
                    split=split,
                )
            )
    if not records:
        raise ValueError(f"No rows found in {metadata_csv}")
    return records


def group_by_slide(records: Iterable[PatchRecord]) -> dict[str, list[PatchRecord]]:
    grouped: dict[str, list[PatchRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.slide_name, []).append(rec)
    for slide_records in grouped.values():
        slide_records.sort(key=lambda r: (r.row, r.col, r.image_path_rel))
    return grouped

