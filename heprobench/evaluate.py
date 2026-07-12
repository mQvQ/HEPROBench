from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_dataset_config
from .io_h5 import read_slide_h5
from .metrics import global_ssim, mae, mse, pearson, psnr
from .schema import group_by_slide, load_channel_names, load_records
from .validate import validate_submission


def _load_target(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with Image.open(path) as img:
        return np.asarray(img)


def evaluate(config_path: str | Path, pred_dir: str | Path, output_csv: str | Path | None = None) -> dict:
    validate_submission(config_path, pred_dir)
    cfg, _ = load_dataset_config(config_path)
    dataset = cfg["dataset"]
    channel_names = load_channel_names(dataset["channel_names"])
    grouped = group_by_slide(load_records(dataset))
    pred_dir = Path(pred_dir)
    rows: list[dict[str, object]] = []

    for slide_name, slide_records in grouped.items():
        payload = read_slide_h5(pred_dir / f"{slide_name}.h5")
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
            for channel_idx, channel in enumerate(channel_names):
                p = patch_pred[..., channel_idx]
                t = target[..., channel_idx]
                rows.append(
                    {
                        "slide_name": slide_name,
                        "row": rec.row,
                        "col": rec.col,
                        "channel": channel,
                        "mae": mae(p, t),
                        "mse": mse(p, t),
                        "pearson": pearson(p, t),
                        "psnr": psnr(p, t),
                        "ssim": global_ssim(p, t),
                    }
                )

    if output_csv is None:
        output_csv = pred_dir / "metrics.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slide_name", "row", "col", "channel", "mae", "mse", "pearson", "psnr", "ssim"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for metric in ["mae", "mse", "pearson", "psnr", "ssim"]:
        values = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        summary[metric] = float(np.nanmean(values))
    summary["n_rows"] = len(rows)
    summary["metrics_csv"] = str(output_csv)
    with (output_csv.parent / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

