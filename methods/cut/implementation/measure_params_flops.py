#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure CUT generator parameter count and forward-pass FLOPs.

Default setting targets the user request:
  - input_nc=output_nc=17
  - img_size=256

FLOPs backend:
  - prefers `fvcore` if installed
  - falls back to PyTorch profiler (`torch.profiler`, with_flops=True)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


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


def _require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyTorch is not available in this Python environment.\n"
            "Run this script from the same conda/venv you use for training/inference (with `torch` installed)."
        ) from e


def _count_params(model: "Any") -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if bool(p.requires_grad):
            trainable += n
    return total, trainable


def _estimate_flops(model: "Any", inputs: Tuple["Any", ...], *, backend: str, device: str) -> tuple[Optional[int], str]:
    _require_torch()
    import torch

    if backend == "none":
        return None, "none"

    do_fvcore = backend in {"auto", "fvcore"}
    do_profiler = backend in {"auto", "profiler"}

    model.eval()
    with torch.inference_mode():
        if do_fvcore:
            try:
                from fvcore.nn import FlopCountAnalysis  # type: ignore

                flops = int(FlopCountAnalysis(model, inputs).total())
                return flops, "fvcore"
            except Exception:
                if backend == "fvcore":
                    raise

        if do_profiler:
            try:
                from torch.profiler import ProfilerActivity, profile  # type: ignore

                activities = [ProfilerActivity.CPU]
                if str(device).startswith("cuda"):
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities, record_shapes=False, profile_memory=False, with_flops=True) as prof:
                    _ = model(*inputs)
                flops = 0
                for evt in prof.key_averages():
                    v = getattr(evt, "flops", None)
                    if v is None:
                        continue
                    flops += int(v)
                return int(flops), "torch.profiler"
            except Exception:
                if backend == "profiler":
                    raise

    return None, "unavailable"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Measure CUT netG params and FLOPs (forward only).")
    p.add_argument("--device", default="cpu", help="Device for the forward pass (default: cpu)")
    p.add_argument("--img-size", type=int, default=256, help="Input H=W (default: 256)")
    p.add_argument(
        "--input-nc",
        type=int,
        default=None,
        help="Input channels (must equal --output-nc for CUT; optional, default: same as --output-nc)",
    )
    p.add_argument("--output-nc", type=int, default=17, help="Output channels (default: 17)")
    p.add_argument("--netG", default="resnet_9blocks", help="Generator arch (default: resnet_9blocks)")
    p.add_argument("--ngf", type=int, default=64, help="Generator base channels (default: 64)")
    p.add_argument("--normG", default="instance", help="Normalization for G (default: instance)")
    p.add_argument("--use_dropout", action="store_true", help="Enable dropout in G (default: disabled)")
    p.add_argument("--no_antialias", action="store_true", help="Use stride conv downsampling (default: false)")
    p.add_argument("--no_antialias_up", action="store_true", help="Use ConvTranspose upsampling (default: false)")
    p.add_argument("--init_type", default="xavier", help="Init type (default: xavier)")
    p.add_argument("--init_gain", type=float, default=0.02, help="Init gain (default: 0.02)")
    p.add_argument(
        "--flops-backend",
        choices=["auto", "fvcore", "profiler", "none"],
        default="auto",
        help="FLOPs backend preference (default: auto)",
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup forwards before FLOPs profile (default: 1)")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")
    args = p.parse_args(argv)

    _require_torch()
    import torch

    from models import networks  # noqa: E402

    dev = torch.device(str(args.device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.input_nc is not None and int(args.input_nc) != int(args.output_nc):
        raise ValueError("For CUT, require input_nc == output_nc (set --input-nc to match --output-nc, or omit it).")
    input_nc = int(args.output_nc)
    output_nc = int(args.output_nc)

    model = networks.define_G(
        input_nc=int(input_nc),
        output_nc=int(output_nc),
        ngf=int(args.ngf),
        netG=str(args.netG),
        norm=str(args.normG),
        use_dropout=bool(args.use_dropout),
        init_type=str(args.init_type),
        init_gain=float(args.init_gain),
        no_antialias=bool(args.no_antialias),
        no_antialias_up=bool(args.no_antialias_up),
        gpu_ids=[],
        opt=None,
    ).to(dev)
    model.eval()

    x = torch.zeros((1, int(input_nc), int(args.img_size), int(args.img_size)), device=dev, dtype=torch.float32)
    inputs: tuple[Any, ...] = (x,)

    with torch.inference_mode():
        for _ in range(max(0, int(args.warmup))):
            _ = model(*inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    params_total, params_trainable = _count_params(model)

    flops, flops_source = _estimate_flops(model, inputs, backend=str(args.flops_backend), device=str(dev))

    row = {
        "method": "CUT",
        "device": str(dev),
        "img_size": int(args.img_size),
        "input_nc": int(input_nc),
        "output_nc": int(output_nc),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "flops": ("" if flops is None else int(flops)),
        "flops_source": str(flops_source),
    }

    flops_str = "n/a" if flops is None else f"{float(flops) / 1e9:.3f} GFLOPs"
    print(
        f"[ok] CUT out_nc={row['output_nc']} in_nc={row['input_nc']} img={row['img_size']} "
        f"params={float(params_total)/1e6:.3f}M flops={flops_str} ({row['flops_source']})"
    )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "device",
                    "img_size",
                    "input_nc",
                    "output_nc",
                    "params_total",
                    "params_trainable",
                    "flops",
                    "flops_source",
                ],
            )
            w.writeheader()
            w.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
