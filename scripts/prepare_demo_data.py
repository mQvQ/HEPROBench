from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _make_slide_mask(size: int = 512) -> np.ndarray:
    """Create deterministic synthetic cells with slide-global integer IDs."""

    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=np.int32)
    cell_id = 1
    for center_y in range(32, size, 64):
        for center_x in range(32, size, 64):
            radius = 22 + ((cell_id * 7) % 8)
            cell = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
            mask[cell] = cell_id
            cell_id += 1
    return mask


def _cell_annotations(
    slide_name: str,
    mask: np.ndarray,
    target_slide: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values_by_cell: dict[int, np.ndarray] = {}
    split_by_cell: dict[int, str] = {}
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        pixels = mask == cell_id
        ys, _ = np.nonzero(pixels)
        values_by_cell[int(cell_id)] = target_slide[pixels].mean(axis=0)
        split_by_cell[int(cell_id)] = (
            "train" if slide_name == "slide_001" else ("valid" if float(ys.mean()) < mask.shape[0] / 2 else "test")
        )

    # Demo labels are median gates within each split.  This guarantees both
    # classes in valid/test while preserving a transparent intensity-derived
    # ground truth analogous to the real-data GMM gates.
    gates: dict[tuple[str, int], float] = {}
    for split in sorted(set(split_by_cell.values())):
        ids = [cell_id for cell_id, value in split_by_cell.items() if value == split]
        for channel_idx in range(len(CHANNEL_NAMES)):
            gates[(split, channel_idx)] = float(
                np.median([values_by_cell[cell_id][channel_idx] for cell_id in ids])
            )

    for cell_id, values in values_by_cell.items():
        split = split_by_cell[cell_id]
        row: dict[str, object] = {
            "slide_name": slide_name,
            "global_cell_id": cell_id,
            "split": split,
        }
        for channel_idx, channel in enumerate(CHANNEL_NAMES):
            value = float(values[channel_idx])
            row[channel] = value
            row[f"{channel}_pos"] = int(value >= gates[(split, channel_idx)])
        rows.append(row)
    return rows


def generate_demo_data() -> None:
    data_root = REPO_ROOT / "demo_data"
    image_dir = data_root / "images"
    target_dir = data_root / "targets"
    mask_dir = data_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    annotations: list[dict[str, object]] = []
    for slide_idx, slide_name in enumerate(["slide_001", "slide_002"], start=1):
        slide_mask = _make_slide_mask()
        slide_target = np.zeros((512, 512, len(CHANNEL_NAMES)), dtype=np.uint8)
        for row in range(2):
            for col in range(2):
                stem = f"{slide_name}_patch_{row}_{col}"
                image = _make_he_patch(slide_idx, row, col)
                target = _make_target(image, slide_idx, row, col)
                image_rel = f"images/{stem}.jpg"
                target_rel = f"targets/{stem}.npy"
                mask_rel = f"masks/{stem}.npy"
                # Keep the committed JPEG bytes stable across Pillow versions.
                if not (data_root / image_rel).exists():
                    Image.fromarray(image).save(data_root / image_rel, quality=95)
                np.save(data_root / target_rel, target)
                y0, x0 = row * 256, col * 256
                mask_patch = slide_mask[y0 : y0 + 256, x0 : x0 + 256]
                np.save(data_root / mask_rel, mask_patch)
                slide_target[y0 : y0 + 256, x0 : x0 + 256] = target
                rows.append(
                    {
                        "slide_name": slide_name,
                        "row": row,
                        "col": col,
                        "split": "train" if slide_name == "slide_001" else ("valid" if row == 0 else "test"),
                        "image_path": image_rel,
                        "target_path": target_rel,
                        "mask_path": mask_rel,
                    }
                )
        annotations.extend(_cell_annotations(slide_name, slide_mask, slide_target))

    with (data_root / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slide_name", "row", "col", "split", "image_path", "target_path", "mask_path"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    split_dir = data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_fields = ["slide_name", "row", "col", "split", "image_path", "target_path", "mask_path"]
    for split in ("train", "valid", "test"):
        with (split_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
    annotation_fields = ["slide_name", "global_cell_id", "split"]
    for channel in CHANNEL_NAMES:
        annotation_fields.extend([channel, f"{channel}_pos"])
    with (data_root / "cell_annotations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=annotation_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(annotations)
    with (data_root / "channel_names.json").open("w", encoding="utf-8") as f:
        json.dump(CHANNEL_NAMES, f, indent=2)


def generate_checkpoints() -> None:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from methods import build_method

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
