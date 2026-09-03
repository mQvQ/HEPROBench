#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROSIE TMA sliding-window inference -> per-slide HDF5 (infer_protocol.md).

For each TMA core (slide_name):
  1) Stitch HE 256x256 patches into a mosaic.
  2) Slide a window with stride S (default 8) over each patch.
  3) For each SxS block, crop a 128x128 HE region centered on the block
     (from the mosaic with zero padding), resize to 224, run ROSIE, and
     fill the SxS block with the predicted C-vector.
  4) Save per-patch 256x256 predictions into a per-slide H5 file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]  # benchmark/methods/ROSIE
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    h5py = None
    _h5py_import_error = e

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    np = None
    _numpy_import_error = e


_RE_HE_PATCH = re.compile(r"_he_patch_(?P<row>\d+)_(?P<col>\d+)\.(?:jpg|png|tif|tiff)$", re.IGNORECASE)
_RE_CODEX_PATCH = re.compile(r"_codex_patch_(?P<row>\d+)_(?P<col>\d+)\.npy$", re.IGNORECASE)
_RE_GENERIC_PATCH = re.compile(r"_patch_(?P<row>\d+)_(?P<col>\d+)(?:[._]|$)", re.IGNORECASE)
_RE_FOV = re.compile(r"(x_\d+_y_\d+|\[\d+\s*,\s*\d+\])", re.IGNORECASE)
_RE_SUBPATCH_XY = re.compile(r"_x_(?P<x>\d+)_y_(?P<y>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PatchRecord:
    slide_name: str
    row: int
    col: int
    image_path_rel: str
    target_path_rel: str
    image_path_abs: str
    target_path_abs: str


def _require_deps() -> None:
    if np is None:
        raise RuntimeError(f"numpy is required but failed to import: {_numpy_import_error}")
    if h5py is None:
        raise RuntimeError(f"h5py is required but failed to import: {_h5py_import_error}")


def _is_abs(path: str) -> bool:
    return os.path.isabs(path) or (len(path) > 1 and path[1] == ":" and path[0].isalpha())


def _resolve_path(cfg_dir: str, p: Any) -> Any:
    if not isinstance(p, str) or not p.strip():
        return p
    if _is_abs(p):
        return p
    return os.path.join(cfg_dir, p)


def _to_abs(root_dir: str, rel_or_abs: str) -> str:
    if _is_abs(rel_or_abs):
        return rel_or_abs
    return str(Path(root_dir) / rel_or_abs)


def _parse_row_col(image_path_rel: str, target_path_rel: str) -> Tuple[int, int]:
    m = _RE_HE_PATCH.search(os.path.basename(image_path_rel))
    if m:
        return int(m.group("row")), int(m.group("col"))
    m = _RE_CODEX_PATCH.search(os.path.basename(target_path_rel))
    if m:
        return int(m.group("row")), int(m.group("col"))
    m = _RE_GENERIC_PATCH.search(os.path.basename(image_path_rel))
    if m:
        return int(m.group("row")), int(m.group("col"))
    m = _RE_GENERIC_PATCH.search(os.path.basename(target_path_rel))
    if m:
        return int(m.group("row")), int(m.group("col"))
    raise ValueError(f"Cannot parse (row,col) from image/target: {image_path_rel} / {target_path_rel}")


def _parse_fov_name(image_path_rel: str, target_path_rel: str) -> Optional[str]:
    m = _RE_FOV.search(image_path_rel)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    m = _RE_FOV.search(target_path_rel)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    return None


def _parse_subpatch_xy(image_path_rel: str, target_path_rel: str) -> Optional[Tuple[int, int]]:
    m = _RE_SUBPATCH_XY.search(os.path.basename(image_path_rel))
    if m:
        return int(m.group("x")), int(m.group("y"))
    m = _RE_SUBPATCH_XY.search(os.path.basename(target_path_rel))
    if m:
        return int(m.group("x")), int(m.group("y"))
    return None


def _iter_csv_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for row in reader:
            yield row


def _load_channel_names_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("channel_names_file must be a JSON string list")
    return data


def _group_records_by_slide(cfg: Dict[str, Any], csv_path: str, root_dir: str) -> Dict[str, List[PatchRecord]]:
    csv_cols = cfg.get("csv_columns") or {}
    if not isinstance(csv_cols, dict):
        raise ValueError("config.csv_columns must be an object if provided")

    image_col = str(csv_cols.get("image_path") or "image_path")
    target_col = str(csv_cols.get("target_path") or "target_path")
    slide_col = str(csv_cols.get("slide_name") or "slide_name")
    default_slide_name = str(cfg.get("default_slide_name") or "").strip()
    slide_name_from_fov = bool(cfg.get("slide_name_from_fov", False))
    subpatch_from_xy = bool(cfg.get("subpatch_from_xy", False))
    patch_size = int(cfg.get("patch_size", 256))

    slides_raw: Dict[str, List[Tuple[int, int, str, str, str, str, Optional[int], Optional[int]]]] = {}
    for row in _iter_csv_rows(csv_path):
        image_rel = (row.get(image_col) or "").strip()
        target_rel = (row.get(target_col) or "").strip()
        slide_name = (row.get(slide_col) or "").strip()
        if not slide_name and slide_name_from_fov:
            fov = _parse_fov_name(image_rel, target_rel)
            if fov:
                slide_name = fov
        if not slide_name and default_slide_name:
            slide_name = default_slide_name
        if not image_rel or not target_rel or not slide_name:
            msg = f"CSV must have non-empty {image_col},{target_col},{slide_col} for each row"
            if not slide_name and default_slide_name == "":
                msg += f" (HEMIT CSVs often lack {slide_col}; set config.default_slide_name to enable a fallback)"
            raise ValueError(msg)

        r, c = _parse_row_col(image_rel, target_rel)
        image_abs = _to_abs(root_dir, image_rel)
        target_abs = _to_abs(root_dir, target_rel)
        xy = _parse_subpatch_xy(image_rel, target_rel) if subpatch_from_xy else None
        x_off = int(xy[0]) if xy is not None else None
        y_off = int(xy[1]) if xy is not None else None
        slides_raw.setdefault(slide_name, []).append((r, c, image_rel, target_rel, image_abs, target_abs, x_off, y_off))

    slides: Dict[str, List[PatchRecord]] = {}
    for slide_name, items in slides_raw.items():
        factor_x = 1
        factor_y = 1
        if subpatch_from_xy and patch_size > 0:
            sub_rows: List[int] = []
            sub_cols: List[int] = []
            for r, c, _ir, _tr, _ia, _ta, x_off, y_off in items:
                if x_off is None or y_off is None:
                    continue
                if x_off % patch_size != 0 or y_off % patch_size != 0:
                    continue
                sub_cols.append(int(x_off // patch_size))
                sub_rows.append(int(y_off // patch_size))
            factor_x = (max(sub_cols) + 1) if sub_cols else 1
            factor_y = (max(sub_rows) + 1) if sub_rows else 1

        recs: List[PatchRecord] = []
        for r, c, image_rel, target_rel, image_abs, target_abs, x_off, y_off in items:
            rr, cc = int(r), int(c)
            if subpatch_from_xy and patch_size > 0 and x_off is not None and y_off is not None:
                if x_off % patch_size == 0 and y_off % patch_size == 0:
                    rr = rr * factor_y + int(y_off // patch_size)
                    cc = cc * factor_x + int(x_off // patch_size)
            recs.append(
                PatchRecord(
                    slide_name=slide_name,
                    row=rr,
                    col=cc,
                    image_path_rel=image_rel,
                    target_path_rel=target_rel,
                    image_path_abs=image_abs,
                    target_path_abs=target_abs,
                )
            )
        slides[slide_name] = sorted(recs, key=lambda x: (x.row, x.col))
    return slides


def _check_path(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _create_pred_dataset(
    data_grp,
    shape: Tuple[int, int, int, int],
    chunks: Tuple[int, int, int, int],
    cfg: Dict[str, Any],
):
    kwargs = {"shape": shape, "dtype": np.uint8, "chunks": chunks}
    compression = str(cfg.get("compression", "lzf"))
    gzip_level = int(cfg.get("gzip_level", 4))
    if compression == "lzf":
        try:
            return data_grp.create_dataset("pred", compression="lzf", **kwargs)
        except Exception:
            return data_grp.create_dataset("pred", compression="gzip", compression_opts=gzip_level, **kwargs)
    if compression == "gzip":
        return data_grp.create_dataset("pred", compression="gzip", compression_opts=gzip_level, **kwargs)
    if compression == "none":
        return data_grp.create_dataset("pred", **kwargs)
    raise ValueError("config.compression must be one of: lzf, gzip, none")


def _build_mosaic(records: Sequence[PatchRecord], patch_size: int) -> Tuple["np.ndarray", int, int]:
    rows = [r.row for r in records]
    cols = [r.col for r in records]
    n_rows = int(max(rows)) + 1 if rows else 0
    n_cols = int(max(cols)) + 1 if cols else 0
    h = n_rows * patch_size
    w = n_cols * patch_size
    mosaic = np.zeros((h, w, 3), dtype=np.uint8)
    for rec in records:
        if not os.path.exists(rec.image_path_abs):
            raise FileNotFoundError(f"HE patch not found: {rec.image_path_abs}")
        img = Image.open(rec.image_path_abs).convert("RGB")
        if img.size != (patch_size, patch_size):
            raise ValueError(f"Expected {patch_size}x{patch_size} but got {img.size}: {rec.image_path_abs}")
        y0 = rec.row * patch_size
        x0 = rec.col * patch_size
        mosaic[y0 : y0 + patch_size, x0 : x0 + patch_size] = np.asarray(img)
    return mosaic, n_rows, n_cols


def _crop_with_padding(img: "np.ndarray", cy: int, cx: int, crop_size: int) -> "np.ndarray":
    half = crop_size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + crop_size
    x1 = x0 + crop_size

    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(img.shape[0], y1)
    src_x1 = min(img.shape[1], x1)

    crop = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return crop


def _build_transform(img_size: int):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            normalize,
        ]
    )


def _imagenet_normalize_(x: "torch.Tensor") -> "torch.Tensor":
    """
    In-place ImageNet normalization for RGB tensors.
    Expects x in range [0,1], shape [B,3,H,W].
    """
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


def _mosaic_to_device_tensor(mosaic_u8: "np.ndarray", device: torch.device) -> "torch.Tensor":
    """
    Convert HWC uint8 mosaic to CHW float tensor in [0,1] on the target device.
    """
    t = torch.from_numpy(mosaic_u8).permute(2, 0, 1).contiguous()
    t = t.to(device=device, non_blocking=True)
    if device.type == "cuda":
        t = t.to(dtype=torch.float16)
    else:
        t = t.to(dtype=torch.float32)
    return t.div_(255.0)


def _build_grid_base(
    *,
    img_size: int,
    crop_size: int,
    mosaic_h: int,
    mosaic_w: int,
    device: torch.device,
) -> "torch.Tensor":
    """
    Precompute base sampling-grid offsets for `grid_sample` (normalized coords).
    Returns float32 [1, img_size, img_size, 2] where last dim is (x_off_norm, y_off_norm).
    """
    off = (torch.arange(img_size, device=device, dtype=torch.float32) + 0.5 - (img_size / 2.0)) * (
        float(crop_size) / float(img_size)
    )
    off_x_norm = (2.0 * off) / float(mosaic_w)
    off_y_norm = (2.0 * off) / float(mosaic_h)
    gy, gx = torch.meshgrid(off_y_norm, off_x_norm, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def _sample_mosaic_batched(
    *,
    mosaic_chw: "torch.Tensor",
    grid_base: "torch.Tensor",
    centers_yx: "torch.Tensor",
) -> "torch.Tensor":
    """
    Batched crop+resize from a mosaic with zero padding using `grid_sample`.

    mosaic_chw: [3,H,W] float in [0,1]
    grid_base : [1,img,img,2] normalized offsets
    centers_yx: [B,2] (cy,cx) in mosaic pixel coordinates

    Returns: [B,3,img,img]
    """
    _, mosaic_h, mosaic_w = mosaic_chw.shape
    cy = centers_yx[:, 0].to(dtype=torch.float32)
    cx = centers_yx[:, 1].to(dtype=torch.float32)
    center_x = ((2.0 * cx + 1.0) / float(mosaic_w)) - 1.0
    center_y = ((2.0 * cy + 1.0) / float(mosaic_h)) - 1.0
    center = torch.stack([center_x, center_y], dim=-1).view(-1, 1, 1, 2)
    grid = grid_base + center
    # grid_sample requires input and grid to have the same dtype.
    if grid.dtype != mosaic_chw.dtype:
        grid = grid.to(dtype=mosaic_chw.dtype)
    x = mosaic_chw.unsqueeze(0).expand(int(centers_yx.shape[0]), -1, -1, -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if not all(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def _load_model(checkpoint_path: str, num_outputs: int, device: torch.device) -> nn.Module:
    model = models.convnext_small(weights="IMAGENET1K_V1")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    # ROSIE checkpoints include non-tensor metadata that PyTorch 2.6 no longer
    # unpickles by default; opt into the legacy behavior for trusted local files.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        state = _strip_module_prefix(state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[WARN] strict load failed: {e}")
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config file for ROSIE sliding-window inference")
    ap.add_argument("--split", default=None, choices=["valid", "test", "both"], help="Override split")
    ap.add_argument("--csv_path", default=None, help="Override CSV path (template allowed for --split both)")
    ap.add_argument("--batch_size", default=None, type=int, help="Override batch size")
    ap.add_argument("--num_workers", default=None, type=int, help="Override num_workers (unused)")
    ap.add_argument("--gpu_id", default=None, help="Override CUDA_VISIBLE_DEVICES")
    ap.add_argument("--stride", default=None, type=int, help="Override sliding stride")
    ap.add_argument(
        "--type",
        default=None,
        choices=["tma", "wsi"],
        help="tma: one H5 per slide_name (default). wsi: split each slide_name by FOV parsed from paths and save as out_dir/slide_name/x_####_y_####.h5",
    )
    return ap.parse_args()


def main() -> None:
    _require_deps()
    args = get_args()

    cfg_path = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")

    for k in ["csv_path", "root_dir", "out_root", "checkpoint_path", "channel_names_file"]:
        if k in cfg:
            cfg[k] = _resolve_path(cfg_dir, cfg[k])

    if args.csv_path is not None:
        cfg["csv_path"] = _resolve_path(cfg_dir, args.csv_path)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.gpu_id is not None:
        cfg["gpu_id"] = str(args.gpu_id)
    if args.stride is not None:
        cfg["stride"] = int(args.stride)
    if args.type is not None:
        cfg["type"] = str(args.type)

    split_arg = args.split if args.split is not None else cfg.get("split")
    if split_arg is None:
        raise ValueError("Must provide split in config (key 'split') or via CLI --split")
    split_arg = str(split_arg)
    if split_arg not in {"valid", "test", "both"}:
        raise ValueError("split must be 'valid', 'test', or 'both'")

    required = ["checkpoint_path", "root_dir", "out_root", "channel_names_file"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    csv_template = str(cfg.get("csv_path") or "").strip()
    if not csv_template:
        raise ValueError("config.csv_path is required")
    if split_arg == "both" and "{split}" not in csv_template:
        raise ValueError("For --split both with --csv_path, please use a template containing '{split}'")

    if "gpu_id" in cfg and cfg["gpu_id"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_names = _load_channel_names_file(str(cfg["channel_names_file"]))
    num_outputs = len(channel_names)

    patch_size = int(cfg.get("patch_size", 256))
    crop_size = int(cfg.get("crop_size", 128))
    stride = int(cfg.get("stride", 8))
    img_size = int(cfg.get("img_size", 224))
    pred_scale = float(cfg.get("pred_scale", 255.0))

    batch_size = int(cfg.get("batch_size", 64))
    method_name = str(cfg.get("method_name") or "ROSIE")
    root_dir = str(cfg["root_dir"])
    out_root = str(cfg["out_root"])
    checkpoint_path = str(cfg["checkpoint_path"])
    splits_to_run = ["valid", "test"] if split_arg == "both" else [split_arg]

    transform = _build_transform(img_size=img_size)
    model = _load_model(checkpoint_path, num_outputs=num_outputs, device=device)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else None
    str_dt = h5py.string_dtype(encoding="utf-8")
    log_every_patch = int(cfg.get("log_every_patch", 10))
    log_every_batch = int(cfg.get("log_every_batch", 50))

    preprocess_backend_cfg = cfg.get("preprocess_backend")
    if preprocess_backend_cfg is None:
        preprocess_backend_cfg = "grid_sample" if device.type == "cuda" else "pil"

    for split in splits_to_run:
        split_csv = csv_template.format(split=split)
        slides = _group_records_by_slide(cfg, split_csv, root_dir=root_dir)
        slide_items = sorted(slides.items(), key=lambda x: x[0])
        if cfg.get("max_slides") is not None:
            slide_items = slide_items[: int(cfg["max_slides"])]

        if not slide_items:
            raise ValueError(f"No samples found for split={split} csv={split_csv}")

        out_dir = _check_path(os.path.join(out_root, method_name, split))

        print("[CONFIG]")
        print("  split:", split)
        print("  split_csv:", split_csv)
        print("  root_dir:", root_dir)
        print("  out_dir:", out_dir)
        print("  slides:", len(slide_items))
        print("  batch_size:", batch_size)
        print("  stride:", stride)
        print("  crop_size:", crop_size)
        print("  img_size:", img_size)
        print("  C(out):", num_outputs)
        print("  preprocess_backend:", preprocess_backend_cfg)
        print("  log_every_patch:", log_every_patch, "(0=disable)")
        print("  log_every_batch:", log_every_batch, "(0=disable)")

        infer_type = str(cfg.get("type", "tma")).lower()
        if infer_type not in {"tma", "wsi"}:
            raise ValueError("config.type must be one of: tma, wsi")

        for slide_idx, (slide_name, records) in enumerate(slide_items):
            safe_slide = slide_name.replace("/", "_").replace("\\", "_")
            slide_jobs: List[Tuple[Optional[str], List[PatchRecord], str]] = [
                (None, records, os.path.join(out_dir, f"{safe_slide}.h5"))
            ]
            if infer_type == "wsi":
                fov_to_records: Dict[str, List[PatchRecord]] = defaultdict(list)
                for rec in records:
                    fov = _parse_fov_name(rec.image_path_rel, rec.target_path_rel)
                    if not fov:
                        raise ValueError(
                            f"config.type=wsi requires paths to contain x_####_y_####. Missing for slide={slide_name}: {rec.image_path_rel}"
                        )
                    fov_to_records[str(fov)].append(rec)
                slide_dir = _check_path(os.path.join(out_dir, safe_slide))
                slide_jobs = []
                for fov_name, fov_records in sorted(fov_to_records.items(), key=lambda x: x[0]):
                    safe_fov = fov_name.replace("/", "_").replace("\\", "_")
                    slide_jobs.append(
                        (
                            str(fov_name),
                            sorted(fov_records, key=lambda r: (r.row, r.col)),
                            os.path.join(slide_dir, f"{safe_fov}.h5"),
                        )
                    )

            for job_idx, (fov_name, records, out_path) in enumerate(slide_jobs):
                t_slide0 = time.perf_counter()
                if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                    if fov_name is None:
                        print(f"[{slide_idx+1}/{len(slide_items)}] Skip existing: {out_path}")
                    else:
                        print(
                            f"[{slide_idx+1}/{len(slide_items)}][{job_idx+1}/{len(slide_jobs)}] Skip existing: {out_path}"
                        )
                    continue

                mosaic, n_rows, n_cols = _build_mosaic(records, patch_size=patch_size)
                mosaic_h, mosaic_w = int(mosaic.shape[0]), int(mosaic.shape[1])

                preprocess_backend = str(
                    cfg.get("preprocess_backend") or ("grid_sample" if device.type == "cuda" else "pil")
                )
                preprocess_backend = preprocess_backend.strip().lower()
                if preprocess_backend not in {"pil", "grid_sample"}:
                    raise ValueError("config.preprocess_backend must be one of: pil, grid_sample")

                grid_base = None
                mosaic_t = None
                if preprocess_backend == "grid_sample":
                    if device.type != "cuda":
                        preprocess_backend = "pil"
                    else:
                        grid_base = _build_grid_base(
                            img_size=img_size,
                            crop_size=crop_size,
                            mosaic_h=mosaic_h,
                            mosaic_w=mosaic_w,
                            device=device,
                        )
                        try:
                            mosaic_t = _mosaic_to_device_tensor(mosaic, device=device)
                        except RuntimeError as e:
                            print(f"[WARN] failed to move mosaic to GPU; fallback to PIL preprocess: {e}")
                            preprocess_backend = "pil"

                n_steps = (patch_size + stride - 1) // stride
                windows_per_patch = int(n_steps * n_steps)
                print(
                    f"[{slide_idx+1}/{len(slide_items)}] slide={slide_name} patches={len(records)} "
                    f"mosaic={mosaic_h}x{mosaic_w} backend={preprocess_backend} windows/patch={windows_per_patch}"
                )
                if preprocess_backend == "grid_sample" and mosaic_t is not None and grid_base is not None:
                    print(f"  mosaic_t: shape={tuple(mosaic_t.shape)} dtype={mosaic_t.dtype} device={mosaic_t.device}")
                    print(f"  grid_base: shape={tuple(grid_base.shape)} dtype={grid_base.dtype} device={grid_base.device}")

                rows = np.asarray([r.row for r in records], dtype=np.int32)
                cols = np.asarray([r.col for r in records], dtype=np.int32)
                image_paths = np.asarray([r.image_path_rel for r in records], dtype=object)
                target_paths = np.asarray([r.target_path_rel for r in records], dtype=object)
                has_gt = np.asarray([os.path.exists(r.target_path_abs) for r in records], dtype=np.bool_)

                grid_index = np.full((n_rows, n_cols), -1, dtype=np.int64)
                for i, (rr, cc) in enumerate(zip(rows.tolist(), cols.tolist())):
                    if grid_index[rr, cc] != -1:
                        raise ValueError(f"Duplicate (row,col)=({rr},{cc}) in slide={slide_name}")
                    grid_index[rr, cc] = i

                tmp_path = out_path + ".tmp"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                if fov_name is None:
                    print(f"[{slide_idx+1}/{len(slide_items)}] Write: {out_path} (N={len(records)})")
                else:
                    print(f"[{slide_idx+1}/{len(slide_items)}] Write: {out_path} (fov={fov_name}, N={len(records)})")
                with h5py.File(tmp_path, "w") as f:
                    f.attrs["schema_version"] = "heprobench_h5_v1"
                    meta = f.create_group("meta")
                    meta.attrs["schema_version"] = "heprobench_h5_v1"
                    meta.attrs["method_name"] = method_name
                    meta.attrs["split"] = split
                    meta.attrs["slide_name"] = slide_name
                    if fov_name is not None:
                        meta.attrs["fov_name"] = str(fov_name)
                    meta.attrs["patch_size"] = patch_size
                    meta.attrs["stride"] = stride
                    meta.attrs["crop_size"] = crop_size
                    meta.attrs["img_size"] = img_size
                    meta.attrs["pred_scale"] = pred_scale
                    meta.attrs["overlap"] = 0
                    meta.attrs["pred_dtype"] = "uint8"
                    meta.attrs["pred_value_range"] = "0-255"
                    meta.attrs["pred_layout"] = "NHWC"
                    meta.attrs["num_channels"] = int(num_outputs)
                    meta.attrs["gt_format"] = "npy"
                    meta.attrs["gt_path_is_relative"] = True
                    meta.attrs["checkpoint_path"] = checkpoint_path
                    meta.attrs["split_csv"] = split_csv
                    meta.attrs["config_json"] = json.dumps(cfg, ensure_ascii=False)
                    for attr_key, attr_value in meta.attrs.items():
                        f.attrs[attr_key] = attr_value

                    data = f.create_group("data")
                    data.create_dataset("rows", data=rows, dtype=np.int32)
                    data.create_dataset("cols", data=cols, dtype=np.int32)
                    data.create_dataset("grid_index", data=grid_index, dtype=np.int64)
                    data.create_dataset("channel_names", data=np.asarray(channel_names, dtype=object), dtype=str_dt)
                    data.create_dataset("target_paths", data=target_paths, dtype=str_dt)
                    data.create_dataset("image_paths", data=image_paths, dtype=str_dt)
                    data.create_dataset("has_gt", data=has_gt, dtype=np.bool_)

                    pred_shape = (len(records), patch_size, patch_size, int(num_outputs))
                    pred_chunks = (1, patch_size, patch_size, min(int(num_outputs), 8))
                    pred_ds = _create_pred_dataset(data, pred_shape, pred_chunks, cfg)

                    for i, rec in enumerate(records):
                        if log_every_patch > 0 and (i % log_every_patch == 0 or i == len(records) - 1):
                            dt = time.perf_counter() - t_slide0
                            print(
                                f"  [slide {slide_idx+1}/{len(slide_items)}] patch {i+1}/{len(records)} (elapsed {dt:.1f}s)"
                            )

                        patch_pred = np.zeros((patch_size, patch_size, num_outputs), dtype=np.float32)
                        base_y = rec.row * patch_size
                        base_x = rec.col * patch_size

                        windows: List[Tuple[int, int, int, int, int, int]] = []
                        for oy in range(0, patch_size, stride):
                            for ox in range(0, patch_size, stride):
                                block_h = min(stride, patch_size - oy)
                                block_w = min(stride, patch_size - ox)
                                cy = base_y + oy + stride // 2
                                cx = base_x + ox + stride // 2
                                windows.append((cy, cx, oy, ox, block_h, block_w))

                        for w0 in range(0, len(windows), batch_size):
                            batch = windows[w0 : w0 + batch_size]
                            if log_every_batch > 0:
                                bi = w0 // batch_size
                                if bi % log_every_batch == 0:
                                    print(
                                        f"    windows batch {bi+1}/{(len(windows) + batch_size - 1)//batch_size} (B={len(batch)})"
                                    )

                            if preprocess_backend == "grid_sample":
                                assert mosaic_t is not None and grid_base is not None
                                centers = torch.tensor([(cy, cx) for cy, cx, *_ in batch], device=device, dtype=torch.float32)
                                x = _sample_mosaic_batched(mosaic_chw=mosaic_t, grid_base=grid_base, centers_yx=centers)
                                _imagenet_normalize_(x)
                                if device.type == "cuda":
                                    x = x.to(dtype=torch.float16)
                            else:
                                imgs: List[torch.Tensor] = []
                                for cy, cx, _oy, _ox, _bh, _bw in batch:
                                    crop = _crop_with_padding(mosaic, cy=cy, cx=cx, crop_size=crop_size)
                                    img = Image.fromarray(crop)
                                    imgs.append(transform(img))
                                x = torch.stack(imgs, dim=0)
                                if device.type == "cuda":
                                    x = x.to(device, non_blocking=True, dtype=torch.float16)
                                else:
                                    x = x.to(device, non_blocking=True, dtype=torch.float32)

                            if autocast_ctx is None:
                                with torch.inference_mode():
                                    outputs = model(x)
                            else:
                                with torch.inference_mode(), autocast_ctx:
                                    outputs = model(x)

                            preds = outputs.to(dtype=torch.float32).cpu().numpy()
                            for j, (_cy, _cx, oy, ox, bh, bw) in enumerate(batch):
                                patch_pred[oy : oy + bh, ox : ox + bw, :] = preds[j]

                        if pred_scale != 1.0:
                            patch_pred *= pred_scale
                        patch_u8 = np.clip(np.rint(patch_pred), 0, 255).astype(np.uint8)
                        pred_ds[i] = patch_u8

                    if pred_ds.dtype != np.uint8:
                        raise RuntimeError("pred dtype must be uint8")
                    n = len(records)
                    if n > 0:
                        step = max(1, n // min(100, n))
                        for i in range(0, n, step):
                            rr = int(rows[i])
                            cc = int(cols[i])
                            if int(grid_index[rr, cc]) != i:
                                raise RuntimeError("grid_index sanity check failed")

                os.replace(tmp_path, out_path)
                print(f"[{slide_idx+1}/{len(slide_items)}] Done: {out_path} (elapsed {time.perf_counter() - t_slide0:.1f}s)")


if __name__ == "__main__":
    main()
