#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HistoPlexer generator inference peak CUDA memory (batch_size=1) across output dims C.

Assumptions (as requested):
  - input fixed RGB: 3 channels
  - sweep output dim C in [3, 7, 17, 28, 60] by default
  - dummy input size defaults to patch_size (256)

This script measures generator-only inference (G forward), which matches how
`benchmark/methods/HistoPlexer/sp_infer.py` calls the generator during prediction.
"""

import argparse
import csv
import json
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


def _read_base_config(*, base_config: Optional[Path], checkpoint_path: Optional[Path]) -> dict[str, Any]:
    cfg_path: Optional[Path] = None
    if base_config is not None:
        cfg_path = base_config
    elif checkpoint_path is not None:
        cfg_path = checkpoint_path.parent / "config.json"

    if cfg_path is None:
        return {}
    if not cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Base config must be a JSON object: {cfg_path}")
    return data


def _measure_infer_peak(
    *,
    c: int,
    device: str,
    cuda_idx: Optional[int],
    input_size: int,
    patch_size: int,
    use_high_res: bool,
    use_multiscale: bool,
    ngf: int,
    depth: int,
    encoder_padding: int,
    decoder_padding: int,
    fm_feature_size: int,
    warmup: int,
    precision: str,
) -> dict[str, Any]:
    import torch

    from src.models.generator import unet_translator

    dev = torch.device(device)
    g = unet_translator(
        input_nc=3,
        output_nc=c,
        use_high_res=use_high_res,
        use_multiscale=use_multiscale,
        ngf=ngf,
        depth=depth,
        encoder_padding=encoder_padding,
        decoder_padding=decoder_padding,
        extra_feature_size=fm_feature_size,
        device=dev,
    )
    g.eval()

    dtype_in = torch.float16 if (device.startswith("cuda") and precision in {"fp16", "bf16"}) else torch.float32
    x = torch.zeros((1, 3, input_size, input_size), device=dev, dtype=dtype_in)
    extra = None
    if fm_feature_size > 0:
        extra = torch.zeros((1, fm_feature_size), device=dev, dtype=torch.float32)

    if device.startswith("cuda"):
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            for _ in range(max(warmup, 0)):
                y = g(x, extra)[-1]
                # match sp_infer: center crop back to patch_size
                if y.shape[-1] != patch_size:
                    top = (y.shape[-2] - patch_size) // 2
                    left = (y.shape[-1] - patch_size) // 2
                    y = y[:, :, top : top + patch_size, left : left + patch_size]
                _ = y.mean()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    with torch.inference_mode():
        with _autocast_ctx(device, precision):
            y = g(x, extra)[-1]
            if y.shape[-1] != patch_size:
                top = (y.shape[-2] - patch_size) // 2
                left = (y.shape[-1] - patch_size) // 2
                y = y[:, :, top : top + patch_size, left : left + patch_size]
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

    del g, x, extra, y
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
    p = argparse.ArgumentParser(description="Measure HistoPlexer inference peak CUDA memory (input=3, sweep C).")
    p.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    p.add_argument("--channels", type=_split_ints, default="3,7,17,28,60", help="Comma-separated C (default: 3,7,17,28,60)")
    p.add_argument("--warmup", type=int, default=2, help="Warmup forwards before measuring (default: 2)")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16", help="Autocast precision (default: fp16)")
    p.add_argument("--patch-size", type=int, default=256, help="Output patch size (default: 256)")
    p.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Dummy HE input size. Default: patch-size (or he_patch_size if use_high_res).",
    )
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Optional checkpoint .pt to read sibling config.json")
    p.add_argument("--base-config", type=Path, default=None, help="Optional config.json to read architecture defaults")
    p.add_argument("--use-high-res", action="store_true", help="Force use_high_res=True (overrides config)")
    p.add_argument("--use-multiscale", action="store_true", help="Force use_multiscale=True (overrides config)")
    p.add_argument("--ngf", type=int, default=None, help="Override ngf")
    p.add_argument("--depth", type=int, default=None, help="Override depth")
    p.add_argument("--encoder-padding", type=int, default=None, help="Override encoder_padding")
    p.add_argument("--decoder-padding", type=int, default=None, help="Override decoder_padding")
    p.add_argument("--fm-feature-size", type=int, default=None, help="Override fm_feature_size")
    p.add_argument("--out-csv", type=Path, default=Path("histoplexer_infer_peak_memory.csv"), help="Output CSV path")
    args = p.parse_args(argv)

    device, cuda_idx = _parse_device(args.device)

    cfg = _read_base_config(base_config=args.base_config, checkpoint_path=args.checkpoint_path)

    patch_size = int(args.patch_size)
    use_high_res = bool(args.use_high_res) or bool(cfg.get("use_high_res", False))
    use_multiscale = bool(args.use_multiscale) or bool(cfg.get("use_multiscale", True))
    ngf = int(args.ngf if args.ngf is not None else cfg.get("ngf", 32))
    depth = int(args.depth if args.depth is not None else cfg.get("depth", 6))
    encoder_padding = int(args.encoder_padding if args.encoder_padding is not None else cfg.get("encoder_padding", 1))
    decoder_padding = int(args.decoder_padding if args.decoder_padding is not None else cfg.get("decoder_padding", 1))
    fm_feature_size = int(args.fm_feature_size if args.fm_feature_size is not None else cfg.get("fm_feature_size", 0))

    input_size = int(args.input_size) if args.input_size is not None else patch_size
    if use_high_res and args.input_size is None:
        input_size = int(cfg.get("he_patch_size", patch_size))

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
            input_size=input_size,
            patch_size=patch_size,
            use_high_res=use_high_res,
            use_multiscale=use_multiscale,
            ngf=ngf,
            depth=depth,
            encoder_padding=encoder_padding,
            decoder_padding=decoder_padding,
            fm_feature_size=fm_feature_size,
            warmup=int(args.warmup),
            precision=str(args.precision),
        )
        row = {
            "method": "HistoPlexer",
            "phase": "infer",
            "device": device,
            "batch_size": 1,
            "input_nc": 3,
            "output_nc": c,
            "patch_size": patch_size,
            "input_size": input_size,
            "precision": str(args.precision),
            "use_high_res": use_high_res,
            "use_multiscale": use_multiscale,
            "ngf": ngf,
            "depth": depth,
            "encoder_padding": encoder_padding,
            "decoder_padding": decoder_padding,
            "fm_feature_size": fm_feature_size,
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB "
            f"peak_reserved={row['peak_reserved_mb']:.2f}MB out={row['out_shape']} {row['out_dtype']}"
        )

    print("\n=== Summary (HistoPlexer Inference Peak Memory) ===")
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
            "patch_size",
            "input_size",
            "precision",
            "use_high_res",
            "use_multiscale",
            "ngf",
            "depth",
            "encoder_padding",
            "decoder_padding",
            "fm_feature_size",
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

