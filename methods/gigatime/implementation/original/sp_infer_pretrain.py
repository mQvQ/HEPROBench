#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GigaTIME pretrain inference -> benchmark HDF5 output (infer_protocol.md).

This mirrors the benchmark inference protocol used by other methods:
  - group HE patches by slide_name
  - optionally split WSI-style cohorts by FOV parsed from patch paths
  - write per-slide/per-FOV H5 files with uint8 predictions in [0,255]

Model-specific behavior:
  - load the official GigaTIME checkpoint
  - apply the official ImageNet-style normalization from the model config
  - convert output logits to probabilities with sigmoid before quantization
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
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parent
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import archs  # noqa: E402

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
                msg += f" (set config.default_slide_name if the cohort omits {slide_col})"
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


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if not all(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def _unwrap_state(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
        return state
    raise ValueError("checkpoint must be a state_dict or contain model_state_dict/state_dict/model")


def _format_shape(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def _load_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg_path_val = cfg.get("model_config_path")
    if isinstance(cfg_path_val, str) and cfg_path_val.strip():
        cfg_path = str(cfg_path_val)
    else:
        ckpt_path = Path(str(cfg["checkpoint_path"]))
        cfg_path = str(ckpt_path.with_name("config.json"))
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("model_config_path must contain a JSON object")
    return data


def _load_model(
    cfg: Dict[str, Any],
    *,
    num_outputs: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, int, Tuple[float, float, float], Tuple[float, float, float]]:
    model_cfg = _load_model_config(cfg)
    architecture = str(model_cfg.get("architecture") or "gigatime").strip().lower()
    if architecture != "gigatime":
        raise ValueError(f"Unsupported architecture in model config: {architecture}")

    model_args = model_cfg.get("model_args") or {}
    if not isinstance(model_args, dict):
        raise ValueError("model_args in model config must be an object")
    pretrained_cfg = model_cfg.get("pretrained_cfg") or {}
    if not isinstance(pretrained_cfg, dict):
        raise ValueError("pretrained_cfg in model config must be an object")

    input_channels = int(model_args.get("in_chans", 3))
    model_img_size = int(model_args.get("img_size", 256))
    mean = tuple(float(x) for x in pretrained_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in pretrained_cfg.get("std", [0.229, 0.224, 0.225]))

    model = archs.gigatime(num_classes=int(num_outputs), input_channels=input_channels).to(device)

    raw_state = torch.load(str(cfg["checkpoint_path"]), map_location="cpu")
    state = _strip_module_prefix(_unwrap_state(raw_state))

    model_state = model.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    unexpected_keys: List[str] = []
    mismatched_keys: List[str] = []
    for key, value in state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if _format_shape(model_state[key]) != _format_shape(value):
            mismatched_keys.append(f"{key}: ckpt{_format_shape(value)} != model{_format_shape(model_state[key])}")
            continue
        filtered_state[key] = value

    if not filtered_state:
        raise ValueError("No checkpoint tensors matched the GigaTIME model.")

    incompatible = model.load_state_dict(filtered_state, strict=False)
    if incompatible.missing_keys:
        print("  missing_tensors:", len(incompatible.missing_keys))
    if unexpected_keys:
        print("  unexpected_tensors:", len(unexpected_keys))
    if mismatched_keys:
        print("  mismatched_tensors:", len(mismatched_keys))

    print("[LOAD]")
    print("  architecture:", architecture)
    print("  checkpoint_path:", str(cfg["checkpoint_path"]))
    print("  model_config_path:", str(cfg.get("model_config_path") or Path(str(cfg["checkpoint_path"])).with_name("config.json")))
    print("  matched_tensors:", len(filtered_state))
    print("  configured_output_channels:", int(num_outputs))
    print("  model_img_size:", model_img_size)
    print("  mean:", mean)
    print("  std:", std)

    model.eval()
    return model, model_img_size, mean, std


def _load_patch_tensor(
    image_path: str,
    *,
    patch_size: int,
    model_img_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> "np.ndarray":
    img = Image.open(image_path).convert("RGB")
    if img.size != (patch_size, patch_size):
        raise ValueError(f"Expected {patch_size}x{patch_size} patch but got {img.size}: {image_path}")
    if model_img_size != patch_size:
        img = img.resize((model_img_size, model_img_size), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))


def _write_h5(
    *,
    out_path: str,
    records: List[PatchRecord],
    slide_name: str,
    fov_name: Optional[str],
    prefix: str,
    split: str,
    split_csv: str,
    cfg: Dict[str, Any],
    method_name: str,
    channel_names: List[str],
    device: torch.device,
    model: torch.nn.Module,
    model_img_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    patch_size: int,
    pred_scale: float,
    batch_size: int,
) -> None:
    rows = np.asarray([r.row for r in records], dtype=np.int32)
    cols = np.asarray([r.col for r in records], dtype=np.int32)
    image_paths = np.asarray([r.image_path_rel for r in records], dtype=object)
    target_paths = np.asarray([r.target_path_rel for r in records], dtype=object)
    has_gt = np.asarray([os.path.exists(r.target_path_abs) for r in records], dtype=np.bool_)

    n_rows = int(np.max(rows)) + 1 if rows.size > 0 else 0
    n_cols = int(np.max(cols)) + 1 if cols.size > 0 else 0
    grid_index = np.full((n_rows, n_cols), -1, dtype=np.int64)
    for i, (rr, cc) in enumerate(zip(rows.tolist(), cols.tolist())):
        if grid_index[rr, cc] != -1:
            extra = f" fov={fov_name}" if fov_name else ""
            raise ValueError(f"Duplicate (row,col)=({rr},{cc}) in slide={slide_name}{extra}")
        grid_index[rr, cc] = i

    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    str_dt = h5py.string_dtype(encoding="utf-8")
    t0 = time.perf_counter()
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
        meta.attrs["stride"] = patch_size
        meta.attrs["crop_size"] = model_img_size
        meta.attrs["img_size"] = model_img_size
        meta.attrs["norm"] = "gigatime_pretrained_cfg"
        meta.attrs["pred_scale"] = pred_scale
        meta.attrs["pred_activation"] = "sigmoid"
        meta.attrs["overlap"] = 0
        meta.attrs["pred_dtype"] = "uint8"
        meta.attrs["pred_value_range"] = "0-255"
        meta.attrs["pred_layout"] = "NHWC"
        meta.attrs["num_channels"] = int(len(channel_names))
        meta.attrs["gt_format"] = "npy"
        meta.attrs["gt_path_is_relative"] = True
        meta.attrs["checkpoint_path"] = str(cfg["checkpoint_path"])
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

        pred_shape = (len(records), patch_size, patch_size, int(len(channel_names)))
        pred_chunks = (1, patch_size, patch_size, min(int(len(channel_names)), 8))
        pred_ds = _create_pred_dataset(data, pred_shape, pred_chunks, cfg)

        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else None
        for start in range(0, len(records), batch_size):
            end = min(len(records), start + batch_size)
            batch_records = records[start:end]
            batch_np = np.stack(
                [
                    _load_patch_tensor(
                        rec.image_path_abs,
                        patch_size=patch_size,
                        model_img_size=model_img_size,
                        mean=mean,
                        std=std,
                    )
                    for rec in batch_records
                ],
                axis=0,
            )
            x = torch.from_numpy(batch_np).to(device=device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(dtype=torch.float16)

            if autocast_ctx is None:
                with torch.inference_mode():
                    logits = model(x)
            else:
                with torch.inference_mode(), autocast_ctx:
                    logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits.to(dtype=torch.float32))
            if tuple(int(v) for v in probs.shape[-2:]) != (patch_size, patch_size):
                probs = F.interpolate(probs, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

            pred = probs.permute(0, 2, 3, 1).cpu().numpy()
            if pred_scale != 1.0:
                pred *= pred_scale
            pred_u8 = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
            pred_ds[start:end] = pred_u8

            print(f"{prefix} batch={start//batch_size + 1} patches={end}/{len(records)}")

    os.replace(tmp_path, out_path)
    print(f"{prefix} Done: {out_path} ({time.perf_counter() - t0:.1f}s)")


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config file for GigaTIME pretrain inference")
    ap.add_argument("--split", default=None, choices=["valid", "test", "both"], help="Override split")
    ap.add_argument("--csv_path", default=None, help="Override CSV path (template allowed for --split both)")
    ap.add_argument("--batch_size", default=None, type=int, help="Override batch size")
    ap.add_argument("--num_workers", default=None, type=int, help="Reserved for compatibility")
    ap.add_argument("--gpu_id", default=None, help="Override CUDA_VISIBLE_DEVICES")
    ap.add_argument("--pred_scale", default=None, type=float, help="Override pred_scale")
    ap.add_argument(
        "--type",
        default=None,
        choices=["tma", "wsi"],
        help="tma: one H5 per slide_name. wsi: split each slide_name by FOV parsed from paths.",
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

    for k in ["csv_path", "root_dir", "out_root", "checkpoint_path", "channel_names_file", "model_config_path"]:
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
    if args.pred_scale is not None:
        cfg["pred_scale"] = float(args.pred_scale)
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

    channel_names = _load_channel_names_file(str(cfg["channel_names_file"]))
    patch_size = int(cfg.get("patch_size", 256))
    pred_scale = float(cfg.get("pred_scale", 255.0))
    batch_size = int(cfg.get("batch_size", 64))
    method_name = str(cfg.get("method_name") or "GIGATIME_pretrain")
    root_dir = str(cfg["root_dir"])
    out_root = str(cfg["out_root"])
    infer_type = str(cfg.get("type", "tma")).lower()
    if infer_type not in {"tma", "wsi"}:
        raise ValueError("config.type must be one of: tma, wsi")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_img_size, mean, std = _load_model(cfg, num_outputs=len(channel_names), device=device)
    splits_to_run = ["valid", "test"] if split_arg == "both" else [split_arg]

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
        print("  patch_size:", patch_size)
        print("  model_img_size:", model_img_size)
        print("  pred_scale:", pred_scale)
        print("  type:", infer_type)
        print("  C(out):", len(channel_names))

        for slide_idx, (slide_name, records) in enumerate(slide_items):
            safe_slide = slide_name.replace("/", "_").replace("\\", "_")
            prefix = f"[{slide_idx+1}/{len(slide_items)}]"

            if infer_type == "wsi":
                fov_to_records: Dict[str, List[PatchRecord]] = defaultdict(list)
                for rec in records:
                    fov = _parse_fov_name(rec.image_path_rel, rec.target_path_rel)
                    if not fov:
                        raise ValueError(
                            f"config.type=wsi requires x_####_y_#### in paths. Missing for slide={slide_name}: {rec.image_path_rel}"
                        )
                    fov_to_records[str(fov)].append(rec)

                slide_dir = _check_path(os.path.join(out_dir, safe_slide))
                fov_items = sorted(fov_to_records.items(), key=lambda x: x[0])
                for fov_idx, (fov_name, fov_records) in enumerate(fov_items):
                    safe_fov = fov_name.replace("/", "_").replace("\\", "_")
                    out_path = os.path.join(slide_dir, f"{safe_fov}.h5")
                    if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                        print(f"{prefix}[{fov_idx+1}/{len(fov_items)}] Skip existing: {out_path}")
                        continue
                    _write_h5(
                        out_path=out_path,
                        records=sorted(fov_records, key=lambda r: (r.row, r.col)),
                        slide_name=slide_name,
                        fov_name=fov_name,
                        prefix=f"{prefix}[{fov_idx+1}/{len(fov_items)}]",
                        split=split,
                        split_csv=split_csv,
                        cfg=cfg,
                        method_name=method_name,
                        channel_names=channel_names,
                        device=device,
                        model=model,
                        model_img_size=model_img_size,
                        mean=mean,
                        std=std,
                        patch_size=patch_size,
                        pred_scale=pred_scale,
                        batch_size=batch_size,
                    )
            else:
                out_path = os.path.join(out_dir, f"{safe_slide}.h5")
                if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                    print(f"{prefix} Skip existing: {out_path}")
                    continue
                _write_h5(
                    out_path=out_path,
                    records=records,
                    slide_name=slide_name,
                    fov_name=None,
                    prefix=prefix,
                    split=split,
                    split_csv=split_csv,
                    cfg=cfg,
                    method_name=method_name,
                    channel_names=channel_names,
                    device=device,
                    model=model,
                    model_img_size=model_img_size,
                    mean=mean,
                    std=std,
                    patch_size=patch_size,
                    pred_scale=pred_scale,
                    batch_size=batch_size,
                )


if __name__ == "__main__":
    main()
