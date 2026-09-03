#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HEX inference peak CUDA memory (batch_size=1) across output dimensions C.

Assumptions (as requested):
  - input is fixed RGB: 3 channels
  - sweep output dimension C in [3, 7, 17, 28, 60] by default
  - dummy input size defaults to 384x384 (matches sp_infer default)

Important:
  HEX's MUSK backbone may try to download weights if not cached.
  This script refuses to run unless `~/.cache/model.safetensors` exists,
  or you provide `--musk-safetensors` to copy into that location.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Optional


_ROOT = Path(__file__).resolve().parent  # benchmark/methods/HEX
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


def _ensure_musk_weights(musk_safetensors: Optional[Path]) -> Path:
    cache_dir = Path.home() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "model.safetensors"
    if musk_safetensors is not None:
        src = musk_safetensors.expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"--musk-safetensors not found: {src}")
        if dest.exists() and dest.samefile(src):
            return dest
        shutil.copy2(src, dest)
        return dest
    if not dest.exists():
        raise FileNotFoundError(
            "MUSK weights not found at ~/.cache/model.safetensors.\n"
            "Provide --musk-safetensors /abs/path/to/model.safetensors to avoid any download."
        )
    return dest


def _measure_hex_infer_peak(
    *,
    c: int,
    device: str,
    img_size: int,
    musk_img_size: int,
    warmup: int,
    precision: str,
) -> dict[str, Any]:
    import torch

    from hex.hex_architecture_fds import CustomModelFDS

    dev = torch.device(device)
    model = CustomModelFDS(visual_output_dim=1024, num_outputs=c, musk_img_size=musk_img_size).to(dev)
    model.eval()
    model.training_status = False

    dtype_in = torch.float16 if (device.startswith("cuda") and precision in {"fp16", "bf16"}) else torch.float32
    x = torch.zeros((1, 3, img_size, img_size), device=dev, dtype=dtype_in)

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            for _ in range(max(warmup, 0)):
                outputs, _raw = model(x, None, epoch=0)
                _ = outputs.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            outputs, _raw = model(x, None, epoch=0)
            _ = outputs.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        peak_alloc = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak_alloc = 0
        peak_reserved = 0

    out_shape = tuple(int(s) for s in outputs.shape)
    out_dtype = str(outputs.dtype)

    del model, x, outputs, _raw
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure HEX inference peak CUDA memory (input=3, sweep C).")
    parser.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    parser.add_argument("--img-size", type=int, default=384, help="Dummy input size H=W (default: 384)")
    parser.add_argument("--musk-img-size", type=int, default=384, help="MUSK config img_size (default: 384)")
    parser.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated output dims C (default: 3,7,17,28,60)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup forwards before measuring (default: 2)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Autocast precision (default: fp16)",
    )
    parser.add_argument(
        "--musk-safetensors",
        type=Path,
        default=None,
        help="Optional local model.safetensors to copy to ~/.cache/model.safetensors (avoids download).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("hex_infer_peak_memory.csv"),
        help="Output CSV path (default: hex_infer_peak_memory.csv)",
    )
    args = parser.parse_args(argv)

    device, cuda_idx = _parse_device(args.device)
    _ensure_musk_weights(args.musk_safetensors)

    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)

    rows: list[dict[str, Any]] = []
    for c in args.channels:
        c = int(c)
        result = _measure_hex_infer_peak(
            c=c,
            device=device,
            img_size=int(args.img_size),
            musk_img_size=int(args.musk_img_size),
            warmup=int(args.warmup),
            precision=str(args.precision),
        )
        row = {
            "method": "HEX",
            "phase": "infer",
            "device": device,
            "img_size": int(args.img_size),
            "musk_img_size": int(args.musk_img_size),
            "batch_size": 1,
            "input_nc": 3,
            "output_dim": c,
            "precision": str(args.precision),
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB "
            f"peak_reserved={row['peak_reserved_mb']:.2f}MB out={row['out_shape']} {row['out_dtype']}"
        )

    print("\n=== Summary (HEX Inference Peak Memory) ===")
    table_cols = ["C", "precision", "peak_allocated_mb", "peak_reserved_mb", "out_dtype"]
    formatted = [
        [
            str(r["output_dim"]),
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
            "musk_img_size",
            "batch_size",
            "input_nc",
            "output_dim",
            "precision",
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

