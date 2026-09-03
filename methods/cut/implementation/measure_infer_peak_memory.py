#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure CUT generator inference peak CUDA memory for batch_size=1 across channel counts.

Setting (as requested):
  - input_nc = output_nc = C
  - batch_size = 1
  - H = W = 256 (configurable)
  - sweep C in [3, 7, 17, 28, 60] by default

This script measures generator-only inference (netG forward), which matches how
`benchmark/methods/CUT/sp_infer.py` calls `model.netG(x)` during prediction.

Example:
  python benchmark/methods/CUT/measure_infer_peak_memory.py --device cuda:0
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import networks  # noqa: E402


def _split_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024.0 * 1024.0)


def _parse_device(device: str) -> tuple[str, list[int]]:
    dev = device.strip().lower()
    if dev in {"cpu"}:
        return "cpu", []
    if dev.startswith("cuda:"):
        idx = int(dev.split(":", 1)[1])
        return f"cuda:{idx}", [idx]
    if dev.startswith("cuda"):
        return "cuda:0", [0]
    # allow plain integer gpu id
    if dev.isdigit():
        idx = int(dev)
        return f"cuda:{idx}", [idx]
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


def _measure_netg_infer_peak(
    *,
    netg,
    device: str,
    img_size: int,
    channels: int,
    warmup: int,
    precision: str,
) -> dict[str, Any]:
    import torch

    dev = torch.device(device)
    netg = netg.to(dev)
    netg.eval()

    x = torch.zeros((1, channels, img_size, img_size), device=dev, dtype=torch.float32)

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            for _ in range(max(warmup, 0)):
                y = netg(x)
                _ = y.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            y = netg(x)
            _ = y.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        peak_alloc = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak_alloc = 0
        peak_reserved = 0

    return {
        "baseline_allocated_mb": _mb(baseline_alloc),
        "baseline_reserved_mb": _mb(baseline_reserved),
        "peak_allocated_mb": _mb(peak_alloc),
        "peak_reserved_mb": _mb(peak_reserved),
        "out_shape": str(tuple(int(s) for s in y.shape)),
        "out_dtype": str(y.dtype),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure CUT netG inference peak CUDA memory.")
    parser.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    parser.add_argument("--img-size", type=int, default=256, help="H=W (default: 256)")
    parser.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated channel counts C (default: 3,7,17,28,60)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup forwards before measuring (default: 2)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Autocast precision (default: fp16)",
    )
    parser.add_argument("--netG", default="resnet_9blocks", help="Generator arch (default: resnet_9blocks)")
    parser.add_argument("--ngf", type=int, default=64, help="Generator base channels (default: 64)")
    parser.add_argument("--normG", default="instance", help="Normalization for G (default: instance)")
    parser.add_argument("--use_dropout", action="store_true", help="Enable dropout in G (default: disabled)")
    parser.add_argument("--no_antialias", action="store_true", help="Use stride conv downsampling (default: false)")
    parser.add_argument("--no_antialias_up", action="store_true", help="Use ConvTranspose upsampling (default: false)")
    parser.add_argument("--init_type", default="xavier", help="Init type (default: xavier)")
    parser.add_argument("--init_gain", type=float, default=0.02, help="Init gain (default: 0.02)")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("cut_infer_peak_memory.csv"),
        help="Output CSV path (default: cut_infer_peak_memory.csv)",
    )
    args = parser.parse_args(argv)

    device, gpu_ids = _parse_device(args.device)

    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(gpu_ids[0])

    rows: list[dict[str, Any]] = []
    for c in args.channels:
        c = int(c)
        netg = networks.define_G(
            input_nc=c,
            output_nc=c,
            ngf=int(args.ngf),
            netG=str(args.netG),
            norm=str(args.normG),
            use_dropout=bool(args.use_dropout),
            init_type=str(args.init_type),
            init_gain=float(args.init_gain),
            no_antialias=bool(args.no_antialias),
            no_antialias_up=bool(args.no_antialias_up),
            gpu_ids=gpu_ids,
            opt=None,
        )
        result = _measure_netg_infer_peak(
            netg=netg,
            device=device,
            img_size=int(args.img_size),
            channels=c,
            warmup=int(args.warmup),
            precision=str(args.precision),
        )
        row = {
            "method": "CUT",
            "phase": "infer",
            "device": device,
            "img_size": int(args.img_size),
            "batch_size": 1,
            "input_nc": c,
            "output_nc": c,
            "precision": str(args.precision),
            "netG": str(args.netG),
            "ngf": int(args.ngf),
            "normG": str(args.normG),
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB "
            f"peak_reserved={row['peak_reserved_mb']:.2f}MB out={row['out_shape']} {row['out_dtype']}"
        )

        del netg
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n=== Summary (CUT Inference Peak Memory, netG only) ===")
    table_cols = ["C", "precision", "peak_allocated_mb", "peak_reserved_mb", "out_dtype"]
    formatted = [
        [
            str(r["input_nc"]),
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
            "img_size",
            "batch_size",
            "input_nc",
            "output_nc",
            "precision",
            "netG",
            "ngf",
            "normG",
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
