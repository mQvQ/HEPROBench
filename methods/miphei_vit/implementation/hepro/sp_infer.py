#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HEPRO TMA inference -> unified HDF5 output (infer_protocol.md).

This script is based on `HEPRO/run_inference.py` + `HEPRO/src/inference.py`, but
instead of saving per-tile TIFFs, it writes per-slide `.h5` files in the common
schema used by benchmark evaluation.

H5 schema (per slide):
  /meta (attrs): schema_version="heprobench_h5_v1", method_name, split, slide_name, patch_size, ...
  /data:
    pred        uint8 [N,256,256,C]  (NHWC)
    rows/cols   int32 [N]
    grid_index  int64 [n_rows,n_cols]
    channel_names string [C]
    target_paths  string [N]  (relative paths)
    image_paths   string [N]  (relative paths)
    has_gt        bool   [N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataset import NormalizationLayer, get_effective_width_height, get_input_mean_std, get_width_height
from src.generators import get_generator
from src.inference import (
    get_generator_state_dict,
    resize_embed_hemit_statedict,
    validate_load_info,
)

try:
    import albumentations as A  # type: ignore
except Exception as e:  # pragma: no cover
    A = None
    _alb_import_error = e

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


@dataclass(frozen=True)
class PatchRecord:
    slide_name: str
    row: int
    col: int
    image_path_rel: str
    target_path_rel: str
    image_path_abs: str
    target_path_abs: str


_RE_HE_PATCH = re.compile(r"_he_patch_(?P<row>\d+)_(?P<col>\d+)\.(?:jpg|png|tif|tiff)$", re.IGNORECASE)
_RE_CODEX_PATCH = re.compile(r"_codex_patch_(?P<row>\d+)_(?P<col>\d+)\.npy$", re.IGNORECASE)
_RE_GENERIC_PATCH = re.compile(r"_patch_(?P<row>\d+)_(?P<col>\d+)(?:[._]|$)", re.IGNORECASE)
_RE_FOV = re.compile(r"(x_\d+_y_\d+)", re.IGNORECASE)
_RE_SLIDE_HINT = re.compile(r"(\[\d+\s*,\s*\d+\])", re.IGNORECASE)
_RE_SUBPATCH_XY = re.compile(r"_x_(?P<x>\d+)_y_(?P<y>\d+)", re.IGNORECASE)


def _require_deps() -> None:
    if np is None:
        raise RuntimeError(f"numpy is required but failed to import: {_numpy_import_error}")
    if h5py is None:
        raise RuntimeError(f"h5py is required but failed to import: {_h5py_import_error}")
    if A is None:
        raise RuntimeError(f"albumentations is required but failed to import: {_alb_import_error}")


def _is_abs(path: str) -> bool:
    return os.path.isabs(path) or (len(path) > 1 and path[1] == ":" and path[0].isalpha())


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
        return m.group(1)
    m = _RE_FOV.search(target_path_rel)
    if m:
        return m.group(1)
    return None


def _parse_slide_hint(image_path_rel: str, target_path_rel: str) -> Optional[str]:
    m = _RE_SLIDE_HINT.search(image_path_rel)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    m = _RE_SLIDE_HINT.search(target_path_rel)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    return _parse_fov_name(image_path_rel, target_path_rel)


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


def _get_csv_fieldnames(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames)


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
    auto_subpatch_from_xy = bool(cfg.get("auto_subpatch_from_xy", True))
    patch_size = int(cfg.get("patch_size", 256))

    csv_fieldnames = set(_get_csv_fieldnames(csv_path))
    slide_col_present = slide_col in csv_fieldnames

    slides_raw: Dict[str, List[Tuple[int, int, str, str, str, str, Optional[int], Optional[int]]]] = {}
    for row in _iter_csv_rows(csv_path):
        image_rel = (row.get(image_col) or "").strip()
        target_rel = (row.get(target_col) or "").strip()
        slide_name = (row.get(slide_col) or "").strip()
        if not slide_name:
            if not slide_col_present or slide_name_from_fov:
                hint = _parse_slide_hint(image_rel, target_rel)
                if hint:
                    slide_name = hint
        if not slide_name and default_slide_name:
            slide_name = default_slide_name
        if not image_rel or not target_rel or not slide_name:
            msg = f"CSV must have non-empty {image_col},{target_col},{slide_col} for each row"
            if not slide_name and default_slide_name == "":
                msg += f" (HEMIT CSVs often lack {slide_col}; slide name can be inferred from paths like '[rid0,rid1]')"
            raise ValueError(msg)

        r, c = _parse_row_col(image_rel, target_rel)
        image_abs = _to_abs(root_dir, image_rel)
        target_abs = _to_abs(root_dir, target_rel)
        xy = _parse_subpatch_xy(image_rel, target_rel)
        x_off = int(xy[0]) if xy is not None else None
        y_off = int(xy[1]) if xy is not None else None
        slides_raw.setdefault(slide_name, []).append((r, c, image_rel, target_rel, image_abs, target_abs, x_off, y_off))

    slides: Dict[str, List[PatchRecord]] = {}
    for slide_name, items in slides_raw.items():
        has_dupes = False
        seen: set = set()
        for r, c, *_rest in items:
            key = (int(r), int(c))
            if key in seen:
                has_dupes = True
                break
            seen.add(key)

        use_subpatch = subpatch_from_xy or (auto_subpatch_from_xy and has_dupes)
        factor_x = 1
        factor_y = 1
        if use_subpatch and patch_size > 0:
            sub_rows: List[int] = []
            sub_cols: List[int] = []
            for _r, _c, _ir, _tr, _ia, _ta, x_off, y_off in items:
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
            if use_subpatch and patch_size > 0 and x_off is not None and y_off is not None:
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


class HEPRORGBPatchDataset(Dataset):
    def __init__(
        self,
        records: List[PatchRecord],
        patch_size: int,
        spatial_augmentations,
        preprocess_input_fn,
    ):
        self.records = records
        self.patch_size = patch_size
        self.spatial_augmentations = spatial_augmentations
        self.preprocess_input_fn = preprocess_input_fn

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = np.asarray(Image.open(rec.image_path_abs).convert("RGB"))
        if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
            # keep same behavior as TMA pipeline: require fixed patch size
            raise ValueError(f"Expected {self.patch_size}x{self.patch_size} but got {img.shape[:2]}: {rec.image_path_abs}")

        if self.spatial_augmentations is not None:
            img = self.spatial_augmentations(image=img)["image"]

        if self.preprocess_input_fn is not None:
            img = self.preprocess_input_fn(img)

        if not img.flags.writeable:
            img = img.copy()

        x = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # CHW
        return {"image": x}


def _check_path(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _create_pred_dataset(data_grp, shape: Tuple[int, int, int, int], chunks: Tuple[int, int, int, int], cfg: Dict[str, Any]):
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


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config file for HEPRO unified inference")
    ap.add_argument(
        "--split",
        default=None,
        choices=["train", "valid", "test", "both"],
        help="Override split. Use 'both' to run valid+test without editing the JSON.",
    )
    ap.add_argument(
        "--csv_path",
        default=None,
        help="Override CSV path. For --split both, you can pass a template like '.../{split}_filter_...csv'.",
    )
    ap.add_argument("--batch_size", default=None, type=int, help="Override batch size")
    ap.add_argument("--num_workers", default=None, type=int, help="Override num_workers")
    ap.add_argument("--gpu_id", default=None, help="Override CUDA_VISIBLE_DEVICES")
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

    # resolve common paths relative to config file
    def resolve(p: Any) -> Any:
        if not isinstance(p, str) or not p.strip():
            return p
        if _is_abs(p):
            return p
        return os.path.join(cfg_dir, p)

    for k in ["checkpoint_dir", "dataset_config_path", "out_root", "channel_names_file", "csv_path"]:
        if k in cfg:
            cfg[k] = resolve(cfg[k])

    # CLI overrides (optional)
    if args.csv_path is not None:
        cfg["csv_path"] = resolve(args.csv_path)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.gpu_id is not None:
        cfg["gpu_id"] = str(args.gpu_id)
    if args.type is not None:
        cfg["type"] = str(args.type)

    split_arg = args.split if args.split is not None else cfg.get("split")

    required = ["checkpoint_dir", "dataset_config_path", "out_root"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    if split_arg is None:
        raise ValueError("Must provide split in config (key 'split') or via CLI --split")
    split_arg = str(split_arg)
    if split_arg not in {"train", "valid", "test", "both"}:
        raise ValueError("split must be 'train', 'valid', 'test', or 'both'")

    checkpoint_dir = str(cfg["checkpoint_dir"])
    dataset_config_path = str(cfg["dataset_config_path"])
    config_path = str(Path(checkpoint_dir) / "config.yaml")
    base_cfg = OmegaConf.load(config_path)

    dataset_cfg = OmegaConf.load(dataset_config_path)
    # Copy relevant data keys (same as run_inference.py)
    for key in ["slide_dataframe_path", "train_dataframe_path", "val_dataframe_path", "test_dataframe_path", "root_dir", "channel_stats_path", "targ_channel_names", "cohort"]:
        if key in dataset_cfg.data:
            base_cfg.data[key] = dataset_cfg.data[key]

    # override batch size if provided
    if "batch_size" in cfg and cfg["batch_size"] is not None:
        base_cfg.train["batch_size"] = int(cfg["batch_size"])

    method_name = str(cfg.get("method_name") or "HEPRO")
    out_root = str(cfg["out_root"])
    splits_to_run = ["valid", "test"] if split_arg == "both" else [split_arg]

    root_dir = str(base_cfg.data.root_dir or "")
    if not root_dir:
        raise ValueError("config/data.root_dir is required for TMA patch CSV (relative paths)")

    # channel names: prefer explicit channel_names_file, otherwise use cfg.data.targ_channel_names
    if isinstance(cfg.get("channel_names_file"), str) and str(cfg.get("channel_names_file")).strip():
        channel_names = _load_channel_names_file(str(cfg["channel_names_file"]))
    else:
        channel_names = list(base_cfg.data.targ_channel_names)

    nc_in = 3
    nc_out = len(channel_names)

    # Device / GPU id
    if "gpu_id" in cfg and cfg["gpu_id"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = int(cfg.get("patch_size", 256))

    # Preprocessing / model are identical across splits; initialize them after we see the first split's data.
    spatial_augmentations = None
    preprocess_input_fn = None
    generator = None
    width = height = None

    with open(str(base_cfg.data.channel_stats_path), "r") as f:
        channel_stats = json.load(f)
    channel_stats_rgb = get_input_mean_std(base_cfg, channel_stats["RGB"])
    preprocess_input_fn = NormalizationLayer(channel_stats_rgb, mode="he")

    str_dt = h5py.string_dtype(encoding="utf-8")

    csv_template = str(cfg.get("csv_path") or "").strip()
    if split_arg == "both" and csv_template and "{split}" not in csv_template:
        # Safer than guessing replacements.
        raise ValueError("For --split both with --csv_path, please use a template containing '{split}'")

    for split in splits_to_run:
        out_dir = _check_path(os.path.join(out_root, method_name, split))

        if csv_template:
            split_csv = csv_template.format(split=split)
        else:
            if split == "train":
                split_csv = str(base_cfg.data.train_dataframe_path)
            elif split == "valid":
                split_csv = str(base_cfg.data.val_dataframe_path)
            else:
                split_csv = str(base_cfg.data.test_dataframe_path)

        slides = _group_records_by_slide(cfg, split_csv, root_dir=root_dir)
        slide_items = sorted(slides.items(), key=lambda x: x[0])
        if cfg.get("max_slides") is not None:
            slide_items = slide_items[: int(cfg["max_slides"])]

        if not slide_items:
            raise ValueError(f"No samples found for split={split} csv={split_csv}")

        if spatial_augmentations is None or width is None or height is None:
            first_abs = slide_items[0][1][0].image_path_abs
            tmp_df = {"image_path": [first_abs]}
            import pandas as pd  # local import to keep top-level minimal
            wh_df = pd.DataFrame(tmp_df)
            width, height = get_width_height(wh_df)
            width, height = get_effective_width_height(width, height, train=True)
            spatial_augmentations = A.Compose([A.CenterCrop(width=width, height=height)])

        if generator is None:
            generator = get_generator(base_cfg.model.model_name, int(width), nc_in, nc_out, base_cfg)
            use_safetensors = (Path(checkpoint_dir) / "model.safetensors").exists()
            if use_safetensors:
                from safetensors.torch import load_file

                checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
                state_dict = load_file(checkpoint_path, device="cpu")
                strict_load = False
                print("Loading checkpoint from safetensors:", checkpoint_path)
            else:
                checkpoint_path = str(Path(checkpoint_dir) / "model.weights.ckpt")
                raw = torch.load(checkpoint_path, map_location="cpu")
                if "state_dict" not in raw:
                    raise ValueError(f"Unexpected ckpt format (missing state_dict): {checkpoint_path}")
                state_dict = get_generator_state_dict(raw["state_dict"])
                strict_load = True
                print("Loading checkpoint from ckpt:", checkpoint_path)

            if hasattr(generator, "swinT"):
                state_dict = resize_embed_hemit_statedict(state_dict, generator)

            load_info = generator.load_state_dict(state_dict, strict=strict_load)
            if use_safetensors:
                validate_load_info(load_info)

            generator = generator.to(device)
            generator.eval()

        print("[CONFIG]")
        print("  split:", split)
        print("  split_csv:", split_csv)
        print("  root_dir:", root_dir)
        print("  out_dir:", out_dir)
        print("  slides:", len(slide_items))
        print("  batch_size:", int(base_cfg.train.batch_size))
        print("  num_workers:", int(cfg.get("num_workers", 8)))
        print("  C(out):", nc_out)

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
                if os.path.exists(out_path) and not bool(cfg.get("overwrite", False)):
                    if fov_name is None:
                        print(f"[{slide_idx+1}/{len(slide_items)}] Skip existing: {out_path}")
                    else:
                        print(f"[{slide_idx+1}/{len(slide_items)}][{job_idx+1}/{len(slide_jobs)}] Skip existing: {out_path}")
                    continue

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

                dataset = HEPRORGBPatchDataset(
                    records=records,
                    patch_size=patch_size,
                    spatial_augmentations=spatial_augmentations,
                    preprocess_input_fn=preprocess_input_fn,
                )
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=int(base_cfg.train.batch_size),
                    shuffle=False,
                    drop_last=False,
                    num_workers=int(cfg.get("num_workers", 8)),
                    pin_memory=(device.type == "cuda"),
                )

                tmp_path = out_path + ".tmp"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                if fov_name is None:
                    print(f"[{slide_idx+1}/{len(slide_items)}] Write: {out_path} (N={len(records)})")
                else:
                    print(
                        f"[{slide_idx+1}/{len(slide_items)}][{job_idx+1}/{len(slide_jobs)}] Write: {out_path} "
                        f"(fov={fov_name}, N={len(records)})"
                    )
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
                    meta.attrs["num_channels"] = int(nc_out)
                    meta.attrs["gt_format"] = "npy"
                    meta.attrs["gt_path_is_relative"] = True
                    meta.attrs["checkpoint_dir"] = checkpoint_dir
                    meta.attrs["dataset_config_path"] = dataset_config_path
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

                    pred_shape = (len(records), patch_size, patch_size, int(nc_out))
                    pred_chunks = (1, patch_size, patch_size, min(int(nc_out), 8))
                    pred_ds = _create_pred_dataset(data, pred_shape, pred_chunks, cfg)

                    offset = 0
                    with torch.no_grad():
                        for batch in loader:
                            x = batch["image"].to(device, non_blocking=True)
                            y = generator(x)
                            if isinstance(y, (tuple, list)):
                                y = y[0]
                            y = ((y + 0.9) / 1.8).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                            y = y.permute(0, 2, 3, 1).contiguous()
                            y_np = y.cpu().numpy()
                            bs = int(y_np.shape[0])
                            pred_ds[offset : offset + bs] = y_np
                            offset += bs

                    if offset != len(records):
                        raise RuntimeError(
                            f"Write count mismatch for slide={slide_name}: wrote {offset}, expected {len(records)}"
                        )

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
'''
python sp_infer.py --config /path/to/HEPRO/configs/infer/mt-codex-gigatime.json --split valid 
'''
