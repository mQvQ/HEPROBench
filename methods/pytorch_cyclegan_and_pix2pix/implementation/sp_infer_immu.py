#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycleGAN/pix2pix TCGA inference -> per-slide HDF5 output.

This variant expects CSVs with (image_path, slide_name) only, e.g.
KIRC_patches.csv produced by patch sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
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


_RE_TCGA_PATCH = re.compile(
    r"(?P<row>\d+)_(?P<col>\d+)\.(?:jpg|jpeg|png|tif|tiff|bmp)$", re.IGNORECASE
)
_RE_HE_PATCH = re.compile(r"_he_patch_(?P<row>\d+)_(?P<col>\d+)\.(?:jpg|jpeg|png|tif|tiff)$", re.IGNORECASE)
_RE_GENERIC_PATCH = re.compile(r"_patch_(?P<row>\d+)_(?P<col>\d+)(?:[._]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class PatchRecord:
    slide_name: str
    row: int
    col: int
    image_path_rel: str
    image_path_abs: str
    target_path_rel: str
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
    if not rel_or_abs:
        return ""
    if _is_abs(rel_or_abs):
        return rel_or_abs
    return str(Path(root_dir) / rel_or_abs)


def _parse_row_col_from_name(path: str) -> Optional[Tuple[int, int]]:
    base = os.path.basename(path)
    for regex in (_RE_TCGA_PATCH, _RE_HE_PATCH, _RE_GENERIC_PATCH):
        m = regex.search(base)
        if m:
            return int(m.group("row")), int(m.group("col"))
    stem = Path(base).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return int(parts[-2]), int(parts[-1])
    return None


def _parse_row_col(image_path_rel: str, target_path_rel: str) -> Tuple[int, int]:
    rc = _parse_row_col_from_name(image_path_rel)
    if rc is not None:
        return rc
    if target_path_rel:
        rc = _parse_row_col_from_name(target_path_rel)
        if rc is not None:
            return rc
    raise ValueError(f"Cannot parse (row,col) from image: {image_path_rel}")


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
    slide_col = str(csv_cols.get("slide_name") or "slide_name")
    target_col = str(csv_cols.get("target_path") or "target_path")

    slides: Dict[str, List[PatchRecord]] = {}
    for row in _iter_csv_rows(csv_path):
        image_rel = (row.get(image_col) or "").strip()
        slide_name = (row.get(slide_col) or "").strip()
        target_rel = (row.get(target_col) or "").strip()
        if not image_rel or not slide_name:
            raise ValueError(f"CSV must have non-empty {image_col},{slide_col} for each row")

        r, c = _parse_row_col(image_rel, target_rel)
        image_abs = _to_abs(root_dir, image_rel)
        target_abs = _to_abs(root_dir, target_rel)
        slides.setdefault(slide_name, []).append(
            PatchRecord(
                slide_name=slide_name,
                row=r,
                col=c,
                image_path_rel=image_rel,
                image_path_abs=image_abs,
                target_path_rel=target_rel,
                target_path_abs=target_abs,
            )
        )
    for k in list(slides.keys()):
        slides[k] = sorted(slides[k], key=lambda x: (x.row, x.col))
    return slides


class HEPatchDataset(Dataset):
    def __init__(self, records: List[PatchRecord], crop_size: int, input_nc: int):
        self.records = records
        self.crop_size = int(crop_size)
        self.input_nc = int(input_nc)
        if self.input_nc == 1:
            self.color_mode = "L"
            normalize = transforms.Normalize((0.5,), (0.5,))
        else:
            self.color_mode = "RGB"
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _to_uint8(arr: "np.ndarray") -> "np.ndarray":
        if arr.dtype == np.uint8:
            return arr
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _load_image(self, path: str) -> Image.Image:
        ext = path.split(".")[-1].lower()
        if ext == "npy":
            arr = np.load(path, mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                arr = self._to_uint8(arr)
                return Image.fromarray(arr)
            if arr.ndim == 3:
                if arr.shape[-1] >= 3:
                    arr = arr[..., :3]
                elif arr.shape[-1] == 1:
                    arr = arr[..., 0]
                arr = self._to_uint8(arr)
                return Image.fromarray(arr)
            raise ValueError(f"Unsupported npy array shape: {arr.shape}")
        return Image.open(path)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = self._load_image(rec.image_path_abs).convert(self.color_mode)
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
    args += add("--dataroot", cfg.get("root_dir"))
    args += add("--name", cfg.get("name"))
    args += add("--checkpoints_dir", cfg.get("checkpoints_dir"))
    args += add("--model", cfg.get("model", "cycle_gan"))
    args += add("--dataset_mode", cfg.get("dataset_mode", "aligned"))
    args += add("--direction", cfg.get("direction", "AtoB"))
    args += add("--input_nc", cfg.get("input_nc", 3))
    args += add("--output_nc", cfg.get("output_nc"))
    args += add("--netG", cfg.get("netG", "resnet_9blocks"))
    args += add("--norm", cfg.get("norm", "instance"))
    args += add("--preprocess", cfg.get("preprocess", "crop"))
    args += add("--crop_size", cfg.get("crop_size", 256))
    args += add("--load_size", cfg.get("load_size", cfg.get("crop_size", 256)))
    args += add("--batch_size", cfg.get("batch_size", 16))
    args += add("--num_threads", cfg.get("num_threads", 4))
    args += add("--gpu_ids", cfg.get("gpu_ids", "0"))
    args += add("--epoch", cfg.get("epoch", "latest"))
    args += add("--phase", cfg.get("phase", "infer"))

    if cfg.get("ngf") is not None:
        args += add("--ngf", cfg.get("ngf"))
    if cfg.get("ndf") is not None:
        args += add("--ndf", cfg.get("ndf"))
    if cfg.get("netD") is not None:
        args += add("--netD", cfg.get("netD"))
    if cfg.get("n_layers_D") is not None:
        args += add("--n_layers_D", cfg.get("n_layers_D"))
    if cfg.get("init_type") is not None:
        args += add("--init_type", cfg.get("init_type"))
    if cfg.get("init_gain") is not None:
        args += add("--init_gain", cfg.get("init_gain"))
    if cfg.get("panel_key") is not None:
        args += add("--panel_key", cfg.get("panel_key"))
    if cfg.get("load_iter") is not None:
        args += add("--load_iter", cfg.get("load_iter"))
    if cfg.get("suffix") is not None:
        args += add("--suffix", cfg.get("suffix"))
    if cfg.get("no_dropout"):
        args += add("--no_dropout")

    args += add("--serial_batches")
    args += add("--no_flip")
    if cfg.get("eval", True):
        args += add("--eval")

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]] + args
    try:
        opt = TestOptions().parse()
    finally:
        sys.argv = old_argv
    return opt


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config file for CycleGAN/pix2pix TCGA inference")
    ap.add_argument("--split", default=None, help="Override split name (default: config.split or 'all')")
    ap.add_argument("--csv_path", default=None, help="Override CSV path (template allowed for --split both)")
    ap.add_argument("--batch_size", default=None, type=int, help="Override batch size")
    ap.add_argument("--num_workers", default=None, type=int, help="Override num_workers")
    ap.add_argument("--gpu_ids", default=None, help="Override gpu_ids (e.g. 0 or 0,1)")
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
        cfg["num_threads"] = int(args.num_workers)
    if args.gpu_ids is not None:
        cfg["gpu_ids"] = str(args.gpu_ids)

    split_arg = args.split if args.split is not None else cfg.get("split", "all")
    split_arg = str(split_arg)
    if split_arg not in {"valid", "test", "both"}:
        splits_to_run = [split_arg]
    else:
        splits_to_run = ["valid", "test"] if split_arg == "both" else [split_arg]

    required = ["csv_path", "root_dir", "out_root", "channel_names_file", "checkpoints_dir", "name"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    csv_template = str(cfg.get("csv_path") or "").strip()
    if split_arg == "both" and "{split}" not in csv_template:
        raise ValueError("For --split both with --csv_path, please use a template containing '{split}'")

    channel_names = _load_channel_names_file(str(cfg["channel_names_file"]))
    output_nc = int(cfg.get("output_nc", len(channel_names)))
    input_nc = int(cfg.get("input_nc", 3))
    if output_nc != len(channel_names):
        raise ValueError("output_nc must match channel_names length")
    cfg["output_nc"] = output_nc
    cfg["input_nc"] = input_nc

    model_name = str(cfg.get("model", "cycle_gan")).lower()
    if model_name not in {"cycle_gan", "pix2pix"}:
        raise ValueError("model must be one of: cycle_gan, pix2pix")

    opt = _build_test_opt(cfg)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    device = model.device
    direction = str(cfg.get("direction", "AtoB"))

    crop_size = int(cfg.get("crop_size", 256))
    patch_size = int(cfg.get("patch_size", crop_size))

    root_dir = str(cfg["root_dir"])
    out_root = str(cfg["out_root"])
    method_name = str(cfg.get("method_name") or model_name)
    str_dt = h5py.string_dtype(encoding="utf-8")

    if model_name == "cycle_gan":
        net_g = model.netG_A if direction == "AtoB" else model.netG_B
    else:
        net_g = model.netG

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
        print("  model:", model_name)

        for slide_idx, (slide_name, records) in enumerate(slide_items):
            safe_slide = slide_name.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(out_dir, f"{safe_slide}.h5")
            if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                print(f"[{slide_idx+1}/{len(slide_items)}] Skip existing: {out_path}")
                continue
            _write_h5(
                out_path=out_path,
                records=records,
                slide_name=slide_name,
                split=split,
                split_csv=split_csv,
                root_dir=root_dir,
                cfg=cfg,
                crop_size=crop_size,
                patch_size=patch_size,
                output_nc=output_nc,
                input_nc=input_nc,
                channel_names=channel_names,
                str_dt=str_dt,
                net_g=net_g,
                model=model,
                device=device,
            )


def _write_h5(
    *,
    out_path: str,
    records: List[PatchRecord],
    slide_name: str,
    split: str,
    split_csv: str,
    root_dir: str,
    cfg: Dict[str, Any],
    crop_size: int,
    patch_size: int,
    output_nc: int,
    input_nc: int,
    channel_names: List[str],
    str_dt,
    net_g,
    model,
    device: torch.device,
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
            raise ValueError(f"Duplicate (row,col)=({rr},{cc}) in slide={slide_name}")
        grid_index[rr, cc] = i

    dataset = HEPatchDataset(records=records, crop_size=crop_size, input_nc=input_nc)
    loader = DataLoader(
        dataset=dataset,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.get("num_threads", 4)),
        pin_memory=(device.type == "cuda"),
    )

    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"Write: {out_path} (slide={slide_name}, N={len(records)})")

    with h5py.File(tmp_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "heprobench_h5_v1"
        meta.attrs["method_name"] = str(cfg.get("method_name") or str(cfg.get("model", "cycle_gan")))
        meta.attrs["split"] = split
        meta.attrs["slide_name"] = slide_name
        meta.attrs["patch_size"] = patch_size
        meta.attrs["overlap"] = 0
        meta.attrs["pred_dtype"] = "uint8"
        meta.attrs["pred_value_range"] = "0-255"
        meta.attrs["pred_layout"] = "NHWC"
        meta.attrs["num_channels"] = int(output_nc)
        meta.attrs["gt_format"] = str(cfg.get("gt_format") or ("npy" if bool(has_gt.any()) else "none"))
        meta.attrs["gt_path_is_relative"] = True
        meta.attrs["checkpoints_dir"] = str(cfg["checkpoints_dir"])
        meta.attrs["experiment_name"] = str(cfg["name"])
        meta.attrs["epoch"] = str(cfg.get("epoch", "latest"))
        meta.attrs["split_csv"] = split_csv
        meta.attrs["config_json"] = json.dumps(cfg, ensure_ascii=False)

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
                y = net_g(x)
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
