#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HistoPlexer generator end-to-end inference time: CPU -> GPU -> CPU (no patch I/O).

Timed path includes (matching sp_infer.py behavior, but without reading/writing patches):
  - H2D copy (image + optional extra_features)
  - model forward (use last output only)
  - center-crop to patch_size
  - clamp->uint8 scaling and NHWC permute
  - D2H copy + numpy conversion

Default:
  - img_size=256 (input patch size)
  - patch_size=256 (output crop size)
  - bsz=1, iters=64, warmup=5
  - sweep output_nc in [3, 7, 17, 28, 60]
  - use_high_res=False (required for img_size=256 parity)
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

from src.models.generator import unet_translator  # noqa: E402


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


def _center_crop_tensor(x, size: int):
    _, _, h, w = x.shape
    if h == size and w == size:
        return x
    if h < size or w < size:
        raise ValueError(f"Model output {h}x{w} smaller than expected {size}")
    top = (h - size) // 2
    left = (w - size) // 2
    return x[:, :, top : top + size, left : left + size]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Measure HistoPlexer e2e inference time (CPU->GPU->CPU), excluding patch I/O."
    )
    p.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu)")
    p.add_argument("--img-size", type=int, default=256, help="Input H=W (default: 256)")
    p.add_argument("--patch-size", type=int, default=256, help="Output crop size H=W (default: 256)")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    p.add_argument("--iters", type=int, default=64, help="Measured iterations (default: 64)")
    p.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated output channel counts C (default: 3,7,17,28,60)",
    )
    p.add_argument("--input-nc", type=int, default=3, help="Input channels (default: 3)")
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Autocast precision (default: fp16)",
    )
    p.add_argument("--use-high-res", action="store_true", help="Enable use_high_res (default: false)")
    p.add_argument(
        "--use-multiscale",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable use_multiscale (default: true)",
    )
    p.add_argument("--ngf", type=int, default=32, help="Base channels (default: 32)")
    p.add_argument("--depth", type=int, default=6, help="Depth (default: 6)")
    p.add_argument("--encoder-padding", type=int, default=1, help="Encoder padding (default: 1)")
    p.add_argument("--decoder-padding", type=int, default=1, help="Decoder padding (default: 1)")
    p.add_argument("--extra-feature-size", type=int, default=0, help="Extra feature size (default: 0)")
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

    import torch

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)
    dev = torch.device(device)

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin_memory = bool(args.pin_memory) and dev.type == "cuda"
    non_blocking = bool(args.non_blocking) and dev.type == "cuda"

    # use_high_res=True typically assumes higher-res input than output; for img_size=256 parity, keep it off.
    if bool(args.use_high_res) and int(args.img_size) == int(args.patch_size):
        raise ValueError("For img_size=256 parity, use --no-use-high-res (default).")

    x_cpu = torch.zeros(
        (int(args.batch_size), int(args.input_nc), int(args.img_size), int(args.img_size)),
        device="cpu",
        dtype=torch.float32,
        pin_memory=pin_memory,
    )

    extra_cpu = None
    if int(args.extra_feature_size) > 0:
        extra_cpu = torch.zeros(
            (int(args.batch_size), int(args.extra_feature_size)),
            device="cpu",
            dtype=torch.float32,
            pin_memory=pin_memory,
        )

    results: list[dict[str, Any]] = []
    for c in [int(x) for x in args.channels]:
        model = unet_translator(
            input_nc=int(args.input_nc),
            output_nc=int(c),
            use_high_res=bool(args.use_high_res),
            use_multiscale=bool(args.use_multiscale),
            ngf=int(args.ngf),
            depth=int(args.depth),
            encoder_padding=int(args.encoder_padding),
            decoder_padding=int(args.decoder_padding),
            extra_feature_size=int(args.extra_feature_size),
            device=dev,
        )
        model = model.to(dev)
        model.eval()

        def _autocast_ctx():
            import contextlib

            if dev.type != "cuda":
                return contextlib.nullcontext()
            if args.precision == "fp32":
                return contextlib.nullcontext()
            if args.precision == "fp16":
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            if args.precision == "bf16":
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            raise ValueError(f"Unsupported precision: {args.precision!r}")

        def run_once() -> None:
            x = x_cpu.to(dev, non_blocking=non_blocking) if dev.type == "cuda" else x_cpu
            extra = None
            if extra_cpu is not None:
                extra = extra_cpu.to(dev, non_blocking=non_blocking) if dev.type == "cuda" else extra_cpu
            with torch.inference_mode():
                with _autocast_ctx():
                    y = model(x, extra)[-1]
            y = _center_crop_tensor(y, int(args.patch_size))
            y = y.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            y = y.permute(0, 2, 3, 1).contiguous()
            _ = y.to("cpu").numpy()

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

        results.append(
            {
                "method": "HistoPlexer",
                "device": str(dev),
                "precision": str(args.precision),
                "img_size": int(args.img_size),
                "patch_size": int(args.patch_size),
                "batch_size": int(args.batch_size),
                "iters": int(args.iters),
                "warmup": int(args.warmup),
                "input_nc": int(args.input_nc),
                "output_nc": int(c),
                "use_high_res": bool(args.use_high_res),
                "use_multiscale": bool(args.use_multiscale),
                "ngf": int(args.ngf),
                "depth": int(args.depth),
                "encoder_padding": int(args.encoder_padding),
                "decoder_padding": int(args.decoder_padding),
                "extra_feature_size": int(args.extra_feature_size),
                "p50_ms_batch": float(p50),
                "p90_ms_batch": float(p90),
                "p99_ms_batch": float(p99),
                "mean_ms_batch": float(mean_ms_batch),
                "mean_ms_patch": float(mean_ms_patch),
                "total_ms": float(total_ms),
            }
        )

    print("[CONFIG]")
    print("  device:", str(dev))
    print("  precision:", str(args.precision))
    print("  img_size:", int(args.img_size))
    print("  patch_size:", int(args.patch_size))
    print("  batch_size:", int(args.batch_size))
    print("  iters:", int(args.iters))
    print("  input_nc:", int(args.input_nc))
    print("  use_high_res:", bool(args.use_high_res))
    print("  use_multiscale:", bool(args.use_multiscale))
    print("  ngf:", int(args.ngf))
    print("  depth:", int(args.depth))
    print("  extra_feature_size:", int(args.extra_feature_size))
    print("")
    print("[RESULT] e2e time (CPU->GPU->CPU), excluding patch I/O")
    print("  C(out)  p50(ms/b)  p90(ms/b)  p99(ms/b)  mean(ms/b)  mean(ms/p)  total(ms)")
    for r in results:
        print(
            f"  {int(r['output_nc']):>5d}  "
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

