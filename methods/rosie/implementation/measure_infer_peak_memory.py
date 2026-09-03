#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure ROSIE (ConvNeXt) inference peak CUDA memory (batch_size=1) across output dims C.

Assumptions (as requested):
  - input fixed RGB: 3 channels
  - sweep output dim C in [3, 7, 17, 28, 60] by default

This is generator-only style inference: one forward on a dummy tensor.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _split_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024.0 * 1024.0)


def _parse_device(device: str) -> tuple[str, Optional[int]]:
    dev = device.strip().lower()
    if dev in {"cpu"}:
        return "cpu", None
    if dev.startswith("cuda:"):
        idx = int(dev.split(":", 1)[1])
        return f"cuda:{idx}", idx
    if dev == "cuda":
        return "cuda:0", 0
    if dev.isdigit():
        idx = int(dev)
        return f"cuda:{idx}", idx
    raise ValueError(f"Unsupported device format: {device!r} (use cpu, cuda, cuda:N, or N)")


def _autocast_ctx(device: str, precision: str):
    import contextlib
    import torch

    if not device.startswith("cuda"):
        return contextlib.nullcontext()
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported precision: {precision!r}")


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    if any(str(k).startswith("module.") for k in state.keys()):
        return {str(k)[len("module.") :]: v for k, v in state.items()}
    return state


def _filter_state_by_shape(state: Dict[str, Any], model_state: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
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


def _resolve_convnext_weights(tv_weights: str):
    import torchvision.models as models

    if tv_weights == "none":
        return None
    if tv_weights != "imagenet":
        raise ValueError("torchvision_weights must be 'imagenet' or 'none'")
    # Avoid hard-failing on torchvision API differences.
    try:
        return models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    except Exception:
        try:
            return "IMAGENET1K_V1"
        except Exception:
            return None


def _load_rosie_model(*, num_outputs: int, checkpoint_path: Optional[Path], device: Any, tv_weights: str):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    weights = _resolve_convnext_weights(tv_weights)
    model = models.convnext_small(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, int(num_outputs))

    if checkpoint_path is not None:
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
            print(f"[warn] skipped {len(skipped)} keys due to missing/shape mismatch (C sweep): e.g. {skipped[:3]}")
        if load_info.unexpected_keys:
            print(f"[warn] unexpected_keys: {load_info.unexpected_keys[:3]}")

    model = model.to(device)
    model.eval()
    return model


def _measure_infer_peak(
    *,
    c: int,
    device: str,
    cuda_idx: Optional[int],
    batch_size: int,
    img_size: int,
    precision: str,
    warmup: int,
    checkpoint_path: Optional[Path],
    tv_weights: str,
) -> dict[str, Any]:
    import torch

    dev = torch.device(device)
    if device.startswith("cuda"):
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)

    model = _load_rosie_model(num_outputs=c, checkpoint_path=checkpoint_path, device=dev, tv_weights=tv_weights)

    x = torch.zeros((int(batch_size), 3, int(img_size), int(img_size)), device=dev, dtype=torch.float32)

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            for _ in range(max(int(warmup), 0)):
                y = model(x)
                _ = y.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            y = model(x)
            _ = y.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        peak_alloc = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak_alloc = 0
        peak_reserved = 0

    out_shape = tuple(int(s) for s in y.shape)
    out_dtype = str(y.dtype)

    del model, x, y
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "baseline_allocated_mb": _mb(baseline_alloc),
        "baseline_reserved_mb": _mb(baseline_reserved),
        "peak_allocated_mb": _mb(peak_alloc),
        "peak_reserved_mb": _mb(peak_reserved),
        "out_shape": str(out_shape),
        "out_dtype": out_dtype,
    }


def _safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Measure ROSIE inference peak CUDA memory (input=3, sweep C).")
    p.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--img-size", type=int, default=224, help="Model input H=W (default: 224)")
    p.add_argument("--warmup", type=int, default=2, help="Warmup forwards before measuring (default: 2)")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16", help="Autocast precision (default: fp16)")
    p.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated C sweep (default: 3,7,17,28,60)",
    )
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Optional checkpoint to load (shape-mismatch filtered)")
    p.add_argument("--torchvision-weights", choices=["imagenet", "none"], default="none", help="ConvNeXt init (default: none)")
    p.add_argument("--out-csv", type=Path, default=_ROOT / "rosie_infer_peak_memory.csv", help="Output CSV path")
    args = p.parse_args(argv)

    device, cuda_idx = _parse_device(str(args.device))

    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)
        torch.backends.cudnn.benchmark = True

    rows: list[dict[str, Any]] = []
    for c in args.channels:
        c = int(c)
        result = _measure_infer_peak(
            c=c,
            device=device,
            cuda_idx=cuda_idx,
            batch_size=int(args.batch_size),
            img_size=int(args.img_size),
            precision=str(args.precision),
            warmup=int(args.warmup),
            checkpoint_path=args.checkpoint_path,
            tv_weights=str(args.torchvision_weights),
        )

        row = {
            "method": "ROSIE",
            "phase": "infer",
            "device": device,
            "batch_size": int(args.batch_size),
            "input_nc": 3,
            "output_nc": c,
            "img_size": int(args.img_size),
            "precision": str(args.precision),
            "torchvision_weights": str(args.torchvision_weights),
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else "",
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB peak_reserved={row['peak_reserved_mb']:.2f}MB "
            f"out={row['out_shape']} {row['out_dtype']}"
        )

    print("\n=== Summary (ROSIE Inference Peak Memory) ===")
    table_cols = ["C", "precision", "peak_allocated_mb", "peak_reserved_mb", "out_dtype"]
    formatted = [
        [
            str(r["output_nc"]),
            str(r["precision"]),
            f"{float(r['peak_allocated_mb']):.2f}",
            f"{float(r['peak_reserved_mb']):.2f}",
            str(r["out_dtype"]),
        ]
        for r in rows
    ]
    widths = [len(c) for c in table_cols]
    for row in formatted:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(table_cols[i].ljust(widths[i]) for i in range(len(widths))))
    print("  ".join("-" * widths[i] for i in range(len(widths))))
    for row in formatted:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(widths))))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "phase",
            "device",
            "batch_size",
            "input_nc",
            "output_nc",
            "img_size",
            "precision",
            "torchvision_weights",
            "checkpoint_path",
            "baseline_allocated_mb",
            "baseline_reserved_mb",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "out_shape",
            "out_dtype",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"[ok] wrote CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

