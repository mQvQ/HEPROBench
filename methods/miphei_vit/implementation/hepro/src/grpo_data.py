"""Data utilities for GRPO-based post-training on HE->mIF."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except Exception:  # pragma: no cover
    A = None

try:
    from .augmentations import HedColorAugmentor
except Exception:  # pragma: no cover
    HedColorAugmentor = None


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def build_target_channel_indices(
    channel_stats: Dict[str, Dict[str, float]],
    target_channel_names: Sequence[str],
) -> List[int]:
    indices: List[int] = []
    for channel_name in target_channel_names:
        if channel_name not in channel_stats:
            raise KeyError(f"Channel '{channel_name}' not found in channel stats")
        info = channel_stats[channel_name]
        if "idx_channel" in info:
            idx = info["idx_channel"]
        elif "channel_idx" in info:
            idx = info["channel_idx"]
        else:
            raise KeyError(
                f"Channel '{channel_name}' has no 'idx_channel' or 'channel_idx' key: {info}"
            )
        indices.append(int(idx))
    return indices


def derive_nuclei_relpath_from_target(
    target_rel: str,
    if_prefix: str = "if_image/",
    mask_prefix: str = "slide-seg-deepcell-mask-mesmer/",
    target_suffix_regex: str = r"_codex_patch_(\d+_\d+)\.npy$",
    mask_suffix_repl: str = r"_mask_patch_\1.npy",
) -> str:
    out = target_rel.replace(if_prefix, mask_prefix, 1)
    out = re.sub(target_suffix_regex, mask_suffix_repl, out)
    return out


def _ensure_abs(root_dir: Path, path_or_rel: str) -> Path:
    p = Path(path_or_rel)
    if p.is_absolute():
        return p
    return root_dir / p


class ColorRolloutAugmentor:
    """Color-only augmenter for rollout generation."""

    def __init__(self, cfg: Optional[Dict[str, float]], training: bool = True):
        self.training = training
        self.cfg = cfg or {}
        self._albu = None

        if (not self.training) or (A is None):
            return

        hed_thresh = float(self.cfg.get("hed_thresh", 0.015))
        hed_p = float(self.cfg.get("hed_p", 0.25))

        brightness_limit = float(self.cfg.get("brightness_limit", 0.2))
        contrast_limit = float(self.cfg.get("contrast_limit", 0.2))
        brightness_contrast_p = float(self.cfg.get("brightness_contrast_p", 0.5))

        blur_p = float(self.cfg.get("blur_p", 0.1))
        blur_sigma_min = float(self.cfg.get("blur_sigma_min", 0.1))
        blur_sigma_max = float(self.cfg.get("blur_sigma_max", 1.5))

        noise_p = float(self.cfg.get("noise_p", 0.1))
        noise_std_min = float(self.cfg.get("noise_std_min", 0.02))
        noise_std_max = float(self.cfg.get("noise_std_max", 0.05))

        transforms = []
        if HedColorAugmentor is not None:
            transforms.append(HedColorAugmentor(thresh=hed_thresh, p=hed_p))
        transforms.extend(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=brightness_contrast_p,
                ),
                A.GaussianBlur(
                    blur_limit=(7, 7),
                    sigma_limit=(blur_sigma_min, blur_sigma_max),
                    p=blur_p,
                ),
                A.GaussNoise(std_range=(noise_std_min, noise_std_max), p=noise_p),
            ]
        )
        self._albu = A.Compose(transforms)

    def _fallback_augment(self, image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32)
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-15.0, 15.0)
        out = out * alpha + beta
        if np.random.rand() < 0.2:
            noise = np.random.normal(loc=0.0, scale=6.0, size=out.shape)
            out = out + noise
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.training:
            return image
        if self._albu is not None:
            return self._albu(image=image)["image"]
        return self._fallback_augment(image)


class CRCCodexGRPODataset(Dataset):
    """CRC-CODEX patch dataset returning rollout groups and cell masks."""

    def __init__(
        self,
        rows: Iterable[Dict[str, str]],
        root_dir: str,
        targ_channel_idxs: Sequence[int],
        preprocess_input_fn=None,
        preprocess_target_fn=None,
        num_rollouts: int = 4,
        training: bool = True,
        mask_builder_cfg: Optional[Dict[str, str]] = None,
        color_aug_cfg: Optional[Dict[str, float]] = None,
        quality_filter: Optional[Dict[str, float]] = None,
    ):
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be >= 1")

        self.root_dir = Path(root_dir)
        self.rows = list(rows)
        self.targ_channel_idxs = list(targ_channel_idxs)
        self.preprocess_input_fn = preprocess_input_fn
        self.preprocess_target_fn = preprocess_target_fn
        self.num_rollouts = int(num_rollouts)
        self.training = bool(training)
        self.mask_builder_cfg = mask_builder_cfg or {}
        self.quality_filter = quality_filter or {}
        self.rollout_augmentor = ColorRolloutAugmentor(color_aug_cfg, training=self.training)

        min_robust_nmi = self.quality_filter.get("min_robust_nmi", None)
        if min_robust_nmi is not None:
            filtered = []
            threshold = float(min_robust_nmi)
            for row in self.rows:
                v = row.get("robust_nmi_value", None)
                if v is None or v == "":
                    filtered.append(row)
                    continue
                try:
                    if float(v) >= threshold:
                        filtered.append(row)
                except ValueError:
                    filtered.append(row)
            self.rows = filtered

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_mask_relpath(self, row: Dict[str, str]) -> str:
        if row.get("nuclei_path"):
            return str(row["nuclei_path"])

        target_rel = row["target_path"]
        if_prefix = str(self.mask_builder_cfg.get("if_prefix", "if_image/"))
        mask_prefix = str(
            self.mask_builder_cfg.get("mask_prefix", "slide-seg-deepcell-mask-mesmer/")
        )
        target_suffix_regex = str(
            self.mask_builder_cfg.get("target_suffix_regex", r"_codex_patch_(\d+_\d+)\.npy$")
        )
        mask_suffix_repl = str(
            self.mask_builder_cfg.get("mask_suffix_repl", r"_mask_patch_\1.npy")
        )
        return derive_nuclei_relpath_from_target(
            target_rel,
            if_prefix=if_prefix,
            mask_prefix=mask_prefix,
            target_suffix_regex=target_suffix_regex,
            mask_suffix_repl=mask_suffix_repl,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        image_path = _ensure_abs(self.root_dir, row["image_path"])
        target_path = _ensure_abs(self.root_dir, row["target_path"])
        nuclei_rel = self._resolve_mask_relpath(row)
        nuclei_path = _ensure_abs(self.root_dir, nuclei_rel)

        image = np.asarray(Image.open(image_path))
        if image.ndim == 2:
            image = np.repeat(image[..., None], repeats=3, axis=2)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        target = np.load(target_path)
        if target.ndim == 2:
            target = target[..., None]
        if self.targ_channel_idxs:
            target = target[:, :, self.targ_channel_idxs]
        if target.dtype not in [np.float32, np.uint8]:
            target = target.astype(np.float32)

        nuclei = np.load(nuclei_path)
        if nuclei.ndim != 2:
            raise ValueError(f"Expected 2D nuclei mask, got shape {nuclei.shape} at {nuclei_path}")
        nuclei = nuclei.astype(np.int64)

        image_rollouts: List[np.ndarray] = [image]
        for _ in range(self.num_rollouts - 1):
            image_rollouts.append(self.rollout_augmentor(image.copy()))

        image_group_tensors: List[torch.Tensor] = []
        for img in image_rollouts:
            arr = img
            if self.preprocess_input_fn is not None:
                arr = self.preprocess_input_fn(arr)
            else:
                arr = arr.astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
            image_group_tensors.append(tensor)
        image_group = torch.stack(image_group_tensors, dim=0)

        if self.preprocess_target_fn is not None:
            target = self.preprocess_target_fn(target)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1).float()

        nuclei_tensor = torch.from_numpy(np.ascontiguousarray(nuclei)).long()
        tile_name = Path(row["image_path"]).stem

        out: Dict[str, torch.Tensor] = {
            "image_group": image_group,
            "target": target_tensor,
            "nuclei": nuclei_tensor,
            "tile_name": tile_name,
        }
        if "slide_name" in row:
            out["slide_name"] = row["slide_name"]
        return out
