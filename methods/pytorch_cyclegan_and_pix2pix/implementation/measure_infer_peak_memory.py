#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure Pix2Pix/CycleGAN generator inference peak CUDA memory (batch_size=1) across output dims C.

Assumptions (as requested):
  - input fixed RGB: 3 channels
  - sweep output dim C in [3, 7, 17, 28, 60] by default

Notes:
  - Measures generator-only forward (no discriminator, no post-processing).
  - Avoids DataParallel (does not pass gpu_ids into define_G) for a cleaner single-GPU peak.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Optional


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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


def _default_netG(model: str) -> str:
    m = model.strip().lower()
    if m == "pix2pix":
        return "unet_256"
    if m in {"cycle_gan", "cyclegan"}:
        return "resnet_9blocks"
    raise ValueError(f"Unsupported model: {model!r} (use pix2pix or cycle_gan)")


def _default_norm(model: str) -> str:
    m = model.strip().lower()
    if m == "pix2pix":
        return "batch"
    return "instance"


def _measure_infer_peak(
    *,
    c: int,
    device: str,
    cuda_idx: Optional[int],
    img_size: int,
    ngf: int,
    netG: str,
    norm: str,
    use_dropout: bool,
    init_type: str,
    init_gain: float,
    warmup: int,
) -> dict[str, Any]:
    import torch

    from models import networks

    dev = torch.device(device)
    g = networks.define_G(
        input_nc=3,
        output_nc=c,
        ngf=ngf,
        netG=netG,
        norm=norm,
        use_dropout=use_dropout,
        init_type=init_type,
        init_gain=init_gain,
        gpu_ids=[],  # avoid DataParallel for peak-memory measurement
    ).to(dev)
    g.eval()

    x = torch.zeros((1, 3, img_size, img_size), device=dev, dtype=torch.float32)

    if device.startswith("cuda"):
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        for _ in range(max(int(warmup), 0)):
            y = g(x)
            _ = y.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    with torch.inference_mode():
        y = g(x)
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

    del g, x, y
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
    p = argparse.ArgumentParser(description="Measure Pix2Pix/CycleGAN inference peak CUDA memory (input=3, sweep C).")
    p.add_argument("--model", choices=["pix2pix", "cycle_gan"], default="pix2pix", help="Model family (default: pix2pix)")
    p.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    p.add_argument("--channels", type=_split_ints, default="3,7,17,28,60", help="Comma-separated C (default: 3,7,17,28,60)")
    p.add_argument("--img-size", type=int, default=256, help="Input H=W (default: 256)")
    p.add_argument("--warmup", type=int, default=2, help="Warmup forwards before measuring (default: 2)")
    p.add_argument("--ngf", type=int, default=64, help="Generator base channels (default: 64)")
    p.add_argument("--netG", type=str, default=None, help="Generator arch (default depends on --model)")
    p.add_argument("--norm", type=str, default=None, choices=["batch", "instance", "none"], help="Norm (default depends on --model)")
    p.add_argument("--use-dropout", action="store_true", help="Enable dropout in G (default: off unless specified)")
    p.add_argument("--init-type", type=str, default="normal", help="Init type (default: normal)")
    p.add_argument("--init-gain", type=float, default=0.02, help="Init gain (default: 0.02)")
    p.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: <method>_<model>_infer_peak_memory.csv)")

    args = p.parse_args(argv)

    model = str(args.model)
    netG = str(args.netG) if args.netG is not None else _default_netG(model)
    norm = str(args.norm) if args.norm is not None else _default_norm(model)

    device, cuda_idx = _parse_device(str(args.device))
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = _ROOT / f"{model}_infer_peak_memory.csv"

    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)

    rows: list[dict[str, Any]] = []
    for c in args.channels:
        c = int(c)
        result = _measure_infer_peak(
            c=c,
            device=device,
            cuda_idx=cuda_idx,
            img_size=int(args.img_size),
            ngf=int(args.ngf),
            netG=netG,
            norm=norm,
            use_dropout=bool(args.use_dropout),
            init_type=str(args.init_type),
            init_gain=float(args.init_gain),
            warmup=int(args.warmup),
        )
        row = {
            "method": "pytorch-CycleGAN-and-pix2pix",
            "model": model,
            "phase": "infer",
            "device": device,
            "batch_size": 1,
            "input_nc": 3,
            "output_nc": c,
            "img_size": int(args.img_size),
            "ngf": int(args.ngf),
            "netG": netG,
            "norm": norm,
            "use_dropout": bool(args.use_dropout),
            "init_type": str(args.init_type),
            "init_gain": float(args.init_gain),
            **result,
        }
        rows.append(row)
        print(
            f"[ok] {model} C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB "
            f"peak_reserved={row['peak_reserved_mb']:.2f}MB out={row['out_shape']} {row['out_dtype']}"
        )

    print(f"\n=== Summary ({model} Inference Peak Memory) ===")
    table_cols = ["C", "peak_allocated_mb", "peak_reserved_mb", "out_dtype"]
    formatted = [
        [
            str(r["output_nc"]),
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "model",
            "phase",
            "device",
            "batch_size",
            "input_nc",
            "output_nc",
            "img_size",
            "ngf",
            "netG",
            "norm",
            "use_dropout",
            "init_type",
            "init_gain",
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

    print(f"[ok] wrote CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

