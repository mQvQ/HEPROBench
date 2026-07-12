from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from methods import build_method  # noqa: E402


CHANNEL_NAMES = ["DAPI", "CD3", "CD20", "PanCK"]
METHODS = {
    "miphei_vit_tiny.pt": {"name": "miphei_vit", "encoder_name": "hoptimus0", "width": 16},
    "dpt_fm_hoptimus0_tiny.pt": {"name": "dpt_fm", "encoder_name": "hoptimus0", "width": 16},
    "dpt_fm_conch_tiny.pt": {"name": "dpt_fm", "encoder_name": "conch", "width": 16},
    "cut_tiny.pt": {"name": "cut", "width": 16},
    "hex_tiny.pt": {"name": "hex", "width": 16},
    "gigatime_tiny.pt": {"name": "gigatime", "width": 16},
    "pytorch_cyclegan_and_pix2pix_tiny.pt": {
        "name": "pytorch_cyclegan_and_pix2pix",
        "variant": "pix2pix",
        "width": 16,
    },
    "rosie_tiny.pt": {"name": "rosie", "width": 16},
}


def _make_he_patch(slide_idx: int, row: int, col: int, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    base = (xx + yy + 31 * slide_idx + 17 * row + 29 * col) % 256
    eosin = (0.55 * base + 80 + 15 * np.sin(xx / 18.0)).clip(0, 255)
    hematoxylin = (0.45 * (255 - base) + 60 + 20 * np.cos(yy / 21.0)).clip(0, 255)
    tissue = (((xx - 128) ** 2 + (yy - 128) ** 2) < (104 + 8 * slide_idx) ** 2).astype(np.float32)
    r = 235 - 0.55 * hematoxylin * tissue + 0.12 * eosin
    g = 210 - 0.35 * hematoxylin * tissue + 0.20 * eosin
    b = 225 - 0.18 * eosin * tissue + 0.45 * hematoxylin
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def _make_target(image: np.ndarray, slide_idx: int, row: int, col: int) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    nuclei = (1.0 - rgb[..., 2]) * 180.0 + 40.0
    cd3 = (rgb[..., 1] * 120.0 + 50.0 * np.sin((xx + 13 * row) / 24.0) + 35.0)
    cd20 = (rgb[..., 0] * 100.0 + 70.0 * np.cos((yy + 17 * col) / 29.0) + 45.0)
    panck = ((rgb[..., 0] - rgb[..., 2] + 1.0) * 90.0 + 18.0 * slide_idx)
    target = np.stack([nuclei, cd3, cd20, panck], axis=-1)
    return target.clip(0, 255).astype(np.uint8)


def generate_demo_data() -> None:
    data_root = REPO_ROOT / "demo_data"
    image_dir = data_root / "images"
    target_dir = data_root / "targets"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for slide_idx, slide_name in enumerate(["slide_001", "slide_002"], start=1):
        for row in range(2):
            for col in range(2):
                stem = f"{slide_name}_patch_{row}_{col}"
                image = _make_he_patch(slide_idx, row, col)
                target = _make_target(image, slide_idx, row, col)
                image_rel = f"images/{stem}.jpg"
                target_rel = f"targets/{stem}.npy"
                Image.fromarray(image).save(data_root / image_rel, quality=95)
                np.save(data_root / target_rel, target)
                rows.append(
                    {
                        "slide_name": slide_name,
                        "row": row,
                        "col": col,
                        "split": "demo",
                        "image_path": image_rel,
                        "target_path": target_rel,
                    }
                )

    with (data_root / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slide_name", "row", "col", "split", "image_path", "target_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (data_root / "channel_names.json").open("w", encoding="utf-8") as f:
        json.dump(CHANNEL_NAMES, f, indent=2)


def generate_checkpoints() -> None:
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for index, (filename, method_cfg) in enumerate(METHODS.items()):
        torch.manual_seed(2026 + index)
        model = build_method(method_cfg, out_channels=len(CHANNEL_NAMES))
        payload = {
            "state_dict": model.state_dict(),
            "method": method_cfg,
            "note": "Tiny randomly initialized checkpoint for HEPROBench review demo only.",
        }
        torch.save(payload, ckpt_dir / filename)


def main() -> None:
    generate_demo_data()
    generate_checkpoints()
    print(f"Prepared demo data and tiny checkpoints under {REPO_ROOT}")


if __name__ == "__main__":
    main()

