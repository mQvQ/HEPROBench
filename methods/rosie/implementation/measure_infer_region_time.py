#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure ROSIE sliding-window inference time for one region (CPU -> GPU -> CPU), excluding I/O.

This measures a single synthetic HE region (default 2048x2048) which corresponds to
64 patches of 256x256 arranged in an 8x8 grid.

The timed path follows `benchmark/methods/ROSIE/sp_infer.py` logic:
  - sliding windows over each 256x256 patch (stride S)
  - crop_with_padding from a mosaic (here: the region itself)
  - transform (ToTensor -> Resize -> Normalize)
  - H2D, fp16 autocast forward
  - D2H + cpu().numpy()
  - fill patch_pred blocks, pred_scale, uint8 conversion
  - stitch 64 patch_u8 back into region_u8 on CPU

Supports C sweep (3,7,17,28,60). For each C, the model head is resized to C.
If a checkpoint is provided, weights are loaded with shape-mismatch filtering
to allow C sweep (head weights are skipped when shapes don't match).
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _split_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No values")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    xs = sorted(values)
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, idx))]


def _crop_with_padding(img: "Any", cy: int, cx: int, crop_size: int) -> "Any":
    import numpy as np

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
    from torchvision import transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            normalize,
        ]
    )


def _strip_module_prefix(state: Dict[str, "Any"]) -> Dict[str, "Any"]:
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def _filter_state_by_shape(state: Dict[str, "Any"], model_state: Dict[str, "Any"]) -> Tuple[Dict[str, "Any"], list[str]]:
    kept: Dict[str, Any] = {}
    skipped: list[str] = []
    for k, v in state.items():
        if k not in model_state:
            skipped.append(k)
            continue
        try:
            if tuple(v.shape) != tuple(model_state[k].shape):
                skipped.append(k)
                continue
        except Exception:
            skipped.append(k)
            continue
        kept[k] = v
    return kept, skipped


def _load_rosie_model(*, num_outputs: int, checkpoint_path: Optional[str], device: "Any", tv_weights: str) -> "Any":
    import torch
    import torch.nn as nn
    import torchvision.models as models

    weights = None
    if tv_weights == "imagenet":
        weights = "IMAGENET1K_V1"
    elif tv_weights == "none":
        weights = None
    else:
        raise ValueError("tv_weights must be 'imagenet' or 'none'")

    model = models.convnext_small(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, int(num_outputs))

    if checkpoint_path:
        state = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Unsupported checkpoint format: expected state dict or dict with model_state_dict")
        state = _strip_module_prefix(state)
        model_sd = model.state_dict()
        state, skipped = _filter_state_by_shape(state, model_sd)
        load_info = model.load_state_dict(state, strict=False)
        if skipped:
            print(f"[WARN] skipped {len(skipped)} keys due to missing/shape mismatch (C sweep): e.g. {skipped[:3]}")
        if load_info.unexpected_keys:
            print(f"[WARN] unexpected_keys: {load_info.unexpected_keys[:3]}")
        if load_info.missing_keys:
            # For C sweep, classifier missing keys are expected.
            pass

    model = model.to(device)
    model.eval()
    return model


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure ROSIE sliding-window region inference time (no I/O).")
    p.add_argument("--checkpoint-path", default=None, help="Optional checkpoint to load (allows C sweep by skipping head)")
    p.add_argument(
        "--torchvision-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Torchvision backbone init for convnext_small (default: imagenet).",
    )
    p.add_argument("--gpu-id", default=None, help="CUDA_VISIBLE_DEVICES value (optional)")
    p.add_argument("--device", default=None, help="Torch device (e.g. cuda, cuda:0, cpu). Default: cuda if available.")
    p.add_argument("--region-size", type=int, default=2048, help="Region H=W (default: 2048)")
    p.add_argument("--patch-size", type=int, default=256, help="Patch size H=W (default: 256)")
    p.add_argument("--stride", type=int, default=8, help="Sliding stride (default: 8)")
    p.add_argument("--crop-size", type=int, default=128, help="Crop size before resize (default: 128)")
    p.add_argument("--img-size", type=int, default=224, help="Model input size after resize (default: 224)")
    p.add_argument("--pred-scale", type=float, default=255.0, help="Scale applied to preds before uint8 (default: 255)")
    p.add_argument("--batch-size", type=int, default=64, help="Windows batch size (default: 64)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup region runs (default: 1)")
    p.add_argument("--iters", type=int, default=2, help="Timed region runs (default: 2)")
    p.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated C sweep (default: 3,7,17,28,60)",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for synthetic region (default: 0)")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output")
    return p.parse_args()


def main() -> int:
    args = get_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import numpy as np
    import torch
    from PIL import Image

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    region_size = int(args.region_size)
    patch_size = int(args.patch_size)
    if region_size % patch_size != 0:
        raise ValueError("region_size must be divisible by patch_size (e.g. 2048 / 256 = 8)")

    n_rows = region_size // patch_size
    n_cols = region_size // patch_size
    stride = int(args.stride)
    crop_size = int(args.crop_size)
    img_size = int(args.img_size)
    win_steps = list(range(0, patch_size, stride))
    windows_per_patch = len(win_steps) * len(win_steps)

    rng = np.random.default_rng(int(args.seed))
    mosaic = rng.integers(0, 256, size=(region_size, region_size, 3), dtype=np.uint8)
    transform = _build_transform(img_size=img_size)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else None

    results: list[dict[str, Any]] = []
    for c in [int(x) for x in args.channels]:
        model = _load_rosie_model(
            num_outputs=c,
            checkpoint_path=(str(args.checkpoint_path) if args.checkpoint_path else None),
            device=device,
            tv_weights=str(args.torchvision_weights),
        )

        def run_one_region() -> "Any":
            region_u8 = np.zeros((region_size, region_size, c), dtype=np.uint8)
            for rr in range(n_rows):
                for cc in range(n_cols):
                    patch_pred = np.zeros((patch_size, patch_size, c), dtype=np.float32)
                    base_y = rr * patch_size
                    base_x = cc * patch_size

                    windows: list[tuple[int, int, int, int, int, int]] = []
                    for oy in range(0, patch_size, stride):
                        for ox in range(0, patch_size, stride):
                            block_h = min(stride, patch_size - oy)
                            block_w = min(stride, patch_size - ox)
                            cy = base_y + oy + stride // 2
                            cx = base_x + ox + stride // 2
                            windows.append((cy, cx, oy, ox, block_h, block_w))

                    for w0 in range(0, len(windows), int(args.batch_size)):
                        batch = windows[w0 : w0 + int(args.batch_size)]
                        imgs: list[torch.Tensor] = []
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
                            with torch.no_grad():
                                outputs = model(x)
                        else:
                            with torch.no_grad(), autocast_ctx:
                                outputs = model(x)

                        preds = outputs.to(dtype=torch.float32).cpu().numpy()
                        for j, (_cy, _cx, oy, ox, bh, bw) in enumerate(batch):
                            patch_pred[oy : oy + bh, ox : ox + bw, :] = preds[j]

                    if float(args.pred_scale) != 1.0:
                        patch_pred *= float(args.pred_scale)
                    patch_u8 = np.clip(np.rint(patch_pred), 0, 255).astype(np.uint8)
                    y0 = rr * patch_size
                    x0 = cc * patch_size
                    region_u8[y0 : y0 + patch_size, x0 : x0 + patch_size, :] = patch_u8
            return region_u8

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(max(0, int(args.warmup))):
            _ = run_one_region()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        times_ms: list[float] = []
        for _ in range(max(1, int(args.iters))):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = run_one_region()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        p50 = _percentile(times_ms, 50)
        p90 = _percentile(times_ms, 90)
        p99 = _percentile(times_ms, 99)
        total_ms = float(sum(times_ms))
        mean_ms_region = total_ms / float(len(times_ms)) if times_ms else float("nan")
        patches_per_region = int(n_rows * n_cols)
        windows_per_region = int(patches_per_region * windows_per_patch)
        mean_ms_patch = mean_ms_region / float(patches_per_region) if patches_per_region else float("nan")
        mean_ms_window = mean_ms_region / float(windows_per_region) if windows_per_region else float("nan")

        results.append(
            {
                "method": "ROSIE",
                "device": str(device),
                "torchvision_weights": str(args.torchvision_weights),
                "checkpoint_path": str(args.checkpoint_path or ""),
                "region_size": int(region_size),
                "patch_size": int(patch_size),
                "stride": int(stride),
                "crop_size": int(crop_size),
                "img_size": int(img_size),
                "pred_scale": float(args.pred_scale),
                "batch_size": int(args.batch_size),
                "warmup": int(args.warmup),
                "iters": int(args.iters),
                "C": int(c),
                "windows_per_patch": int(windows_per_patch),
                "windows_per_region": int(windows_per_region),
                "p50_ms_region": float(p50),
                "p90_ms_region": float(p90),
                "p99_ms_region": float(p99),
                "mean_ms_region": float(mean_ms_region),
                "mean_ms_patch": float(mean_ms_patch),
                "mean_ms_window": float(mean_ms_window),
            }
        )

    print("[CONFIG]")
    print("  device:", str(device))
    print("  region_size:", int(region_size))
    print("  patch_size:", int(patch_size))
    print("  grid:", f"{n_rows}x{n_cols} (= {n_rows*n_cols} patches)")
    print("  stride:", int(stride))
    print("  crop_size:", int(crop_size))
    print("  img_size:", int(img_size))
    print("  batch_size(windows):", int(args.batch_size))
    print("  warmup(region):", int(args.warmup))
    print("  iters(region):", int(args.iters))
    print("")
    print("[RESULT] e2e time per 2048x2048 region (CPU->GPU->CPU), excluding I/O")
    print("  C  p50(ms/r)  p90(ms/r)  p99(ms/r)  mean(ms/r)  mean(ms/patch)  mean(ms/window)")
    for r in results:
        print(
            f"  {int(r['C']):>2d}  "
            f"{float(r['p50_ms_region']):>9.3f}  {float(r['p90_ms_region']):>9.3f}  {float(r['p99_ms_region']):>9.3f}  "
            f"{float(r['mean_ms_region']):>10.3f}  {float(r['mean_ms_patch']):>13.6f}  {float(r['mean_ms_window']):>13.6f}"
        )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(results[0].keys()) if results else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"[ok] wrote CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

