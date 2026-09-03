#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure CUT netG end-to-end inference time: CPU -> GPU -> CPU (no patch I/O).

Timed path includes:
  - H2D copy
  - netG forward
  - D2H copy
  - sp_infer-style CPU postprocess: numpy conversion + denormalize_to_uint8 + NHWC transpose

Default setting:
  - batch_size=1
  - iters=64 (i.e., 64 sequential patches)
  - warmup=5
  - img_size=256
  - sweep C in [3, 7, 17, 28, 60] with input_nc=output_nc=C
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Optional


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import networks  # noqa: E402


def _split_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_device(device: str) -> tuple[str, list[int]]:
    dev = device.strip().lower()
    if dev in {"cpu"}:
        return "cpu", []
    if dev.startswith("cuda:"):
        idx = int(dev.split(":", 1)[1])
        return f"cuda:{idx}", [idx]
    if dev.startswith("cuda"):
        return "cuda:0", [0]
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


def _denormalize_to_uint8(y_np: "Any") -> "Any":
    import numpy as np

    # y in [-1,1] -> [0,255]
    y = (y_np + 1.0) * 0.5
    y = np.clip(y, 0.0, 1.0)
    y = (y * 255.0).round().astype("uint8")
    return y


def _check_32bit_index_math_safe(*, batch_size: int, c: int, h: int, w: int, device: str) -> None:
    # Some CUDA kernels (e.g., reflect-pad) require tensor.numel() <= 2^31-1.
    if not str(device).startswith("cuda"):
        return
    numel = int(batch_size) * int(c) * int(h) * int(w)
    if numel > (2**31 - 1):
        raise ValueError(
            "Input tensor is too large for some CUDA kernels that use 32-bit indexing "
            f"(numel={numel} > 2^31-1). Reduce --batch-size and/or --img-size and/or --channels."
        )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Measure CUT netG e2e inference time (CPU->GPU->CPU), excluding patch I/O.")
    p.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    p.add_argument("--img-size", type=int, default=256, help="H=W (default: 256)")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    p.add_argument("--iters", type=int, default=64, help="Measured iterations (default: 64)")
    p.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated output channel counts C (default: 3,7,17,28,60)",
    )
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Autocast precision (default: fp16)",
    )
    p.add_argument("--netG", default="resnet_9blocks", help="Generator arch (default: resnet_9blocks)")
    p.add_argument("--ngf", type=int, default=64, help="Generator base channels (default: 64)")
    p.add_argument("--normG", default="instance", help="Normalization for G (default: instance)")
    p.add_argument("--use_dropout", action="store_true", help="Enable dropout in G (default: disabled)")
    p.add_argument("--no_antialias", action="store_true", help="Use stride conv downsampling (default: false)")
    p.add_argument("--no_antialias_up", action="store_true", help="Use ConvTranspose upsampling (default: false)")
    p.add_argument("--init_type", default="xavier", help="Init type (default: xavier)")
    p.add_argument("--init_gain", type=float, default=0.02, help="Init gain (default: 0.02)")
    p.add_argument(
        "--postprocess",
        choices=["sp_infer", "none"],
        default="sp_infer",
        help="Include sp_infer-style CPU postprocess (default: sp_infer)",
    )
    p.add_argument(
        "--pin-memory",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use pinned host memory (default: true)",
    )
    p.add_argument(
        "--non-blocking",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use non_blocking=True for H2D copy (default: true)",
    )
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")
    args = p.parse_args(argv)

    device, gpu_ids = _parse_device(args.device)

    import numpy as np
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(gpu_ids[0])
        torch.backends.cudnn.benchmark = True

    dev = torch.device(device)
    pin_memory = bool(args.pin_memory) and dev.type == "cuda"
    non_blocking = bool(args.non_blocking) and dev.type == "cuda"

    results: list[dict[str, Any]] = []
    for c in [int(x) for x in args.channels]:
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
        netg = netg.to(dev)
        netg.eval()

        _check_32bit_index_math_safe(
            batch_size=int(args.batch_size),
            c=int(c),
            h=int(args.img_size),
            w=int(args.img_size),
            device=str(device),
        )

        x_cpu = torch.zeros(
            (int(args.batch_size), c, int(args.img_size), int(args.img_size)),
            device="cpu",
            dtype=torch.float32,
            pin_memory=pin_memory,
        )

        def run_once() -> None:
            x = x_cpu.to(dev, non_blocking=non_blocking) if dev.type == "cuda" else x_cpu
            with torch.inference_mode():
                with _autocast_ctx(device, str(args.precision)):
                    y = netg(x)
            y_cpu = y.detach().to("cpu")
            if args.postprocess == "sp_infer":
                y_np = y_cpu.numpy()
                y_u8 = _denormalize_to_uint8(y_np)
                _ = np.transpose(y_u8, (0, 2, 3, 1))  # NHWC
            else:
                _ = y_cpu

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        for _ in range(max(0, int(args.warmup))):
            run_once()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        times_ms: list[float] = []
        for _ in range(max(1, int(args.iters))):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            run_once()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        p50 = _percentile(times_ms, 50)
        p90 = _percentile(times_ms, 90)
        p99 = _percentile(times_ms, 99)
        total_ms = float(sum(times_ms))
        mean_ms_batch = total_ms / float(len(times_ms)) if times_ms else float("nan")
        total_batches = int(len(times_ms))
        total_patches = int(int(args.batch_size) * total_batches)
        mean_ms_patch = total_ms / float(total_patches) if total_patches > 0 else float("nan")

        row = {
            "method": "CUT",
            "device": device,
            "precision": str(args.precision),
            "img_size": int(args.img_size),
            "batch_size": int(args.batch_size),
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "postprocess": str(args.postprocess),
            "netG": str(args.netG),
            "ngf": int(args.ngf),
            "normG": str(args.normG),
            "input_nc": c,
            "output_nc": c,
            "p50_ms_batch": float(p50),
            "p90_ms_batch": float(p90),
            "p99_ms_batch": float(p99),
            "mean_ms_batch": float(mean_ms_batch),
            "mean_ms_patch": float(mean_ms_patch),
            "total_ms": float(total_ms),
        }
        results.append(row)

    print("[CONFIG]")
    print("  device:", device)
    print("  precision:", str(args.precision))
    print("  img_size:", int(args.img_size))
    print("  batch_size:", int(args.batch_size))
    print("  iters:", int(args.iters))
    print("  postprocess:", str(args.postprocess))
    print("  netG:", str(args.netG))
    print("  ngf:", int(args.ngf))
    print("  normG:", str(args.normG))
    print("")
    print("[RESULT] e2e time (CPU->GPU->CPU), excluding patch I/O")
    print("  C  p50(ms/b)  p90(ms/b)  p99(ms/b)  mean(ms/b)  mean(ms/p)  total(ms)")
    for r in results:
        print(
            f"  {int(r['output_nc']):>2d}  "
            f"{float(r['p50_ms_batch']):>9.3f}  {float(r['p90_ms_batch']):>9.3f}  {float(r['p99_ms_batch']):>9.3f}  "
            f"{float(r['mean_ms_batch']):>10.3f}  {float(r['mean_ms_patch']):>10.3f}  {float(r['total_ms']):>8.3f}"
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
