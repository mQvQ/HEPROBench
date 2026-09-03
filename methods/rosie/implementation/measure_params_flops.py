#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure ROSIE model parameter count and forward-pass FLOPs.

ROSIE is implemented here as a torchvision `convnext_small` with a resized head.

Default setting targets the user request:
  - input_nc=3
  - output_nc=17
  - img_size=224

By default we use `--torchvision-weights none` to avoid any weight download.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Optional, Tuple


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


def _load_rosie_model(*, num_outputs: int, tv_weights: str) -> "Any":
    _require_torch()
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
    model.eval()
    return model


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Measure ROSIE params and FLOPs (forward only).")
    p.add_argument("--device", default="cpu", help="Device for the forward pass (default: cpu)")
    p.add_argument("--img-size", type=int, default=224, help="Input H=W (default: 224)")
    p.add_argument("--output-nc", type=int, default=17, help="Output dimension C (default: 17)")
    p.add_argument(
        "--torchvision-weights",
        choices=["imagenet", "none"],
        default="none",
        help="Torchvision backbone init (default: none, avoids download).",
    )
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

    dev = torch.device(str(args.device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model = _load_rosie_model(num_outputs=int(args.output_nc), tv_weights=str(args.torchvision_weights)).to(dev)
    model.eval()

    x = torch.zeros((1, 3, int(args.img_size), int(args.img_size)), device=dev, dtype=torch.float32)
    inputs: tuple[Any, ...] = (x,)

    with torch.inference_mode():
        for _ in range(max(0, int(args.warmup))):
            _ = model(*inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    params_total, params_trainable = _count_params(model)

    flops, flops_source = _estimate_flops(model, inputs, backend=str(args.flops_backend), device=str(dev))

    row = {
        "method": "ROSIE",
        "device": str(dev),
        "img_size": int(args.img_size),
        "input_nc": 3,
        "output_nc": int(args.output_nc),
        "torchvision_weights": str(args.torchvision_weights),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "flops": ("" if flops is None else int(flops)),
        "flops_source": str(flops_source),
    }

    flops_str = "n/a" if flops is None else f"{float(flops) / 1e9:.3f} GFLOPs"
    print(
        f"[ok] ROSIE out_nc={row['output_nc']} img={row['img_size']} tv_weights={row['torchvision_weights']} "
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
                    "torchvision_weights",
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
