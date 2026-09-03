#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUT TMA inference -> per-slide HDF5 output (infer_protocol.md).

This script mirrors the HEPRO/DTR H5 schema, but uses CUT generator.
Inputs are HE patches from CSV; outputs are per-slide H5 with
uint8 [N,256,256,C] predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from options.test_options import TestOptions  # noqa: E402
from models import create_model  # noqa: E402

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


class HEPatchDataset(Dataset):
    def __init__(self, records: List[PatchRecord], crop_size: int, input_nc: int):
        self.records = records
        self.crop_size = crop_size
        self.input_nc = input_nc
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec.image_path_abs).convert("RGB")
        x = self.transform(img)
        if self.input_nc != x.shape[0]:
            repeat_times = (self.input_nc + x.shape[0] - 1) // x.shape[0]
            x = x.repeat(repeat_times, 1, 1)[: self.input_nc]
        return {"image": x}


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


def _build_test_opt(cfg: Dict[str, Any]) -> Any:
    def add(flag: str, value: Optional[Any] = None) -> List[str]:
        if value is None:
            return [flag]
        return [flag, str(value)]

    args: List[str] = []
    args += add("--model", cfg.get("model", "cut"))
    args += add("--CUT_mode", cfg.get("CUT_mode"))
    args += add("--name", cfg.get("name"))
    args += add("--checkpoints_dir", cfg.get("checkpoints_dir"))
    args += add("--epoch", cfg.get("epoch", "latest"))
    args += add("--dataset_mode", cfg.get("dataset_mode", "HE2SPREPEAT"))
    args += add("--direction", cfg.get("direction", "AtoB"))
    args += add("--input_nc", cfg.get("input_nc"))
    args += add("--output_nc", cfg.get("output_nc"))
    args += add("--netG", cfg.get("netG", "resnet_9blocks"))
    args += add("--normG", cfg.get("normG", "instance"))
    args += add("--no_dropout")
    args += add("--preprocess", cfg.get("preprocess", "crop"))
    args += add("--crop_size", cfg.get("crop_size", 256))
    args += add("--load_size", cfg.get("load_size", cfg.get("crop_size", 256)))
    args += add("--batch_size", cfg.get("batch_size", 16))
    args += add("--num_threads", cfg.get("num_threads", 4))
    args += add("--gpu_ids", cfg.get("gpu_ids", "0"))
    args += add("--panel_key", cfg.get("panel_key"))
    args += add("--serial_batches")
    args += add("--no_flip")
    if cfg.get("eval", True):
        args += add("--eval")

    cmd_line = " ".join(args)
    return TestOptions(cmd_line=cmd_line).parse()


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config file for CUT unified inference")
    ap.add_argument("--split", default=None, choices=["valid", "test", "both"], help="Override split")
    ap.add_argument("--csv_path", default=None, help="Override CSV path (template allowed for --split both)")
    ap.add_argument("--batch_size", default=None, type=int, help="Override batch size")
    ap.add_argument("--num_workers", default=None, type=int, help="Override num_workers")
    ap.add_argument("--gpu_ids", default=None, help="Override gpu_ids (e.g. 0 or 0,1)")
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

    for k in ["csv_path", "root_dir", "out_root", "channel_names_file", "checkpoints_dir"]:
        if k in cfg:
            cfg[k] = _resolve_path(cfg_dir, cfg[k])

    if args.csv_path is not None:
        cfg["csv_path"] = _resolve_path(cfg_dir, args.csv_path)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.gpu_ids is not None:
        cfg["gpu_ids"] = str(args.gpu_ids)
    if args.type is not None:
        cfg["type"] = str(args.type)

    split_arg = args.split if args.split is not None else cfg.get("split")
    if split_arg is None:
        raise ValueError("Must provide split in config (key 'split') or via CLI --split")
    split_arg = str(split_arg)
    if split_arg not in {"valid", "test", "both"}:
        raise ValueError("split must be 'valid', 'test', or 'both'")

    required = ["csv_path", "root_dir", "out_root", "channel_names_file", "checkpoints_dir", "name"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    csv_template = str(cfg.get("csv_path") or "").strip()
    if split_arg == "both" and "{split}" not in csv_template:
        raise ValueError("For --split both with --csv_path, please use a template containing '{split}'")

    channel_names = _load_channel_names_file(str(cfg["channel_names_file"]))
    output_nc = int(cfg.get("output_nc", len(channel_names)))
    input_nc = int(cfg.get("input_nc", output_nc))
    if output_nc != len(channel_names):
        raise ValueError("output_nc must match channel_names length")

    cfg["output_nc"] = output_nc
    cfg["input_nc"] = input_nc

    opt = _build_test_opt(cfg)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    device = model.device

    crop_size = int(cfg.get("crop_size", 256))
    patch_size = int(cfg.get("patch_size", crop_size))

    root_dir = str(cfg["root_dir"])
    out_root = str(cfg["out_root"])
    method_name = str(cfg.get("method_name") or "CUT")

    splits_to_run = ["valid", "test"] if split_arg == "both" else [split_arg]
    str_dt = h5py.string_dtype(encoding="utf-8")

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
        print("  batch_size:", int(cfg.get("batch_size", 16)))
        print("  input_nc:", input_nc)
        print("  output_nc:", output_nc)

        infer_type = str(cfg.get("type", "tma")).lower()
        if infer_type not in {"tma", "wsi"}:
            raise ValueError("config.type must be one of: tma, wsi")

        for slide_idx, (slide_name, records) in enumerate(slide_items):
            safe_slide = slide_name.replace("/", "_").replace("\\", "_")

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
                fov_items = sorted(fov_to_records.items(), key=lambda x: x[0])
                for fov_idx, (fov_name, fov_records) in enumerate(fov_items):
                    fov_records = sorted(fov_records, key=lambda r: (r.row, r.col))
                    out_path = os.path.join(slide_dir, f"{fov_name}.h5")
                    if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                        print(
                            f"[{slide_idx+1}/{len(slide_items)}][{fov_idx+1}/{len(fov_items)}] Skip existing: {out_path}"
                        )
                        continue
                    _write_h5(
                        out_path=out_path,
                        records=fov_records,
                        slide_name=slide_name,
                        fov_name=fov_name,
                        split=split,
                        split_csv=split_csv,
                        cfg=cfg,
                        device=device,
                        model=model,
                        method_name=method_name,
                        crop_size=crop_size,
                        patch_size=patch_size,
                        output_nc=output_nc,
                        input_nc=input_nc,
                        channel_names=channel_names,
                        str_dt=str_dt,
                    )
            else:
                out_path = os.path.join(out_dir, f"{safe_slide}.h5")
                if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                    print(f"[{slide_idx+1}/{len(slide_items)}] Skip existing: {out_path}")
                    continue
                _write_h5(
                    out_path=out_path,
                    records=records,
                    slide_name=slide_name,
                    fov_name=None,
                    split=split,
                    split_csv=split_csv,
                    cfg=cfg,
                    device=device,
                    model=model,
                    method_name=method_name,
                    crop_size=crop_size,
                    patch_size=patch_size,
                    output_nc=output_nc,
                    input_nc=input_nc,
                    channel_names=channel_names,
                    str_dt=str_dt,
                )


def _write_h5(
    *,
    out_path: str,
    records: List[PatchRecord],
    slide_name: str,
    fov_name: Optional[str],
    split: str,
    split_csv: str,
    cfg: Dict[str, Any],
    device: torch.device,
    model,
    method_name: str,
    crop_size: int,
    patch_size: int,
    output_nc: int,
    input_nc: int,
    channel_names: List[str],
    str_dt,
) -> None:
    rows = np.asarray([r.row for r in records], dtype=np.int32)
    cols = np.asarray([r.col for r in records], dtype=np.int32)
    image_paths = np.asarray([r.image_path_rel for r in records], dtype=object)
    target_paths = np.asarray([r.target_path_rel for r in records], dtype=object)
    has_gt = np.asarray([os.path.exists(r.target_path_abs) for r in records], dtype=np.bool_)

    n_rows = int(rows.max()) + 1 if rows.size else 0
    n_cols = int(cols.max()) + 1 if rows.size else 0
    grid_index = np.full((n_rows, n_cols), -1, dtype=np.int64)
    for i, (rr, cc) in enumerate(zip(rows.tolist(), cols.tolist())):
        if grid_index[rr, cc] != -1:
            extra = f" fov={fov_name}" if fov_name else ""
            raise ValueError(f"Duplicate (row,col)=({rr},{cc}) in slide={slide_name}{extra}")
        grid_index[rr, cc] = i

    dataset = HEPatchDataset(records=records, crop_size=crop_size, input_nc=input_nc)
    loader = DataLoader(
        dataset=dataset,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if fov_name:
        print(f"Write: {out_path} (slide={slide_name}, fov={fov_name}, N={len(records)})")
    else:
        print(f"Write: {out_path} (slide={slide_name}, N={len(records)})")

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
        meta.attrs["overlap"] = 0
        meta.attrs["pred_dtype"] = "uint8"
        meta.attrs["pred_value_range"] = "0-255"
        meta.attrs["pred_layout"] = "NHWC"
        meta.attrs["num_channels"] = int(output_nc)
        meta.attrs["gt_format"] = "npy"
        meta.attrs["gt_path_is_relative"] = True
        meta.attrs["checkpoints_dir"] = str(cfg["checkpoints_dir"])
        meta.attrs["experiment_name"] = str(cfg["name"])
        meta.attrs["epoch"] = str(cfg.get("epoch", "latest"))
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

        pred_shape = (len(records), patch_size, patch_size, int(output_nc))
        pred_chunks = (1, patch_size, patch_size, min(int(output_nc), 8))
        pred_ds = _create_pred_dataset(data, pred_shape, pred_chunks, cfg)

        offset = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                y = model.netG(x)
                y_np = y.detach().cpu().numpy()
                y_u8 = model.denormalize_to_uint8(y_np)
                y_u8 = np.transpose(y_u8, (0, 2, 3, 1))  # NHWC
                bs = int(y_u8.shape[0])
                pred_ds[offset : offset + bs] = y_u8
                offset += bs

        if offset != len(records):
            raise RuntimeError(f"Write count mismatch for slide={slide_name}: wrote {offset}, expected {len(records)}")

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


if __name__ == "__main__":
    main()
