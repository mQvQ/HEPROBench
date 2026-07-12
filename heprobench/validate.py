from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_dataset_config
from .io_h5 import read_slide_h5
from .schema import group_by_slide, load_channel_names, load_records


def _load_target(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with Image.open(path) as img:
        return np.asarray(img)


def validate_data(config_path: str | Path, check_arrays: bool = True) -> dict:
    cfg, _ = load_dataset_config(config_path)
    dataset = cfg["dataset"]
    channel_names = load_channel_names(dataset["channel_names"])
    records = load_records(dataset)
    seen = set()
    for rec in records:
        if not rec.image_path.exists():
            raise FileNotFoundError(f"Missing image: {rec.image_path}")
        if rec.target_path is None or not rec.target_path.exists():
            raise FileNotFoundError(f"Missing target: {rec.target_path}")
        key = (rec.slide_name, rec.row, rec.col)
        if key in seen:
            raise ValueError(f"Duplicated tile coordinate: {key}")
        seen.add(key)
        if check_arrays:
            with Image.open(rec.image_path) as img:
                if img.mode != "RGB":
                    raise ValueError(f"Expected RGB image: {rec.image_path}")
                if img.size[0] != img.size[1]:
                    raise ValueError(f"Expected square image patch: {rec.image_path}")
            target = _load_target(rec.target_path)
            if target.ndim != 3:
                raise ValueError(f"Expected target shape [H,W,C]: {rec.target_path}")
            if target.shape[-1] != len(channel_names):
                raise ValueError(f"Target channel mismatch for {rec.target_path}")
    grouped = group_by_slide(records)
    return {"slides": len(grouped), "patches": len(records), "channels": channel_names}


def validate_submission(config_path: str | Path, pred_dir: str | Path) -> dict:
    cfg, _ = load_dataset_config(config_path)
    dataset = cfg["dataset"]
    channel_names = load_channel_names(dataset["channel_names"])
    records = load_records(dataset)
    grouped = group_by_slide(records)
    pred_dir = Path(pred_dir)
    checked = 0
    for slide_name, slide_records in grouped.items():
        h5_path = pred_dir / f"{slide_name}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing submission file: {h5_path}")
        payload = read_slide_h5(h5_path)
        if payload["channel_names"] != channel_names:
            raise ValueError(f"{h5_path} channel names do not match dataset config")
        pred = payload["pred"]
        if pred.dtype != np.uint8:
            raise ValueError(f"{h5_path} /data/pred must be uint8")
        if pred.ndim != 4 or pred.shape[-1] != len(channel_names):
            raise ValueError(f"{h5_path} /data/pred must have shape [N,H,W,C]")
        coords = list(zip(payload["rows"].tolist(), payload["cols"].tolist()))
        expected_coords = [(r.row, r.col) for r in slide_records]
        if sorted(coords) != sorted(expected_coords):
            raise ValueError(f"{h5_path} tile coordinates do not match metadata")
        checked += 1
    return {"slides": checked, "channels": channel_names, "pred_dir": str(pred_dir)}

