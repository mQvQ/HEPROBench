#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HistoPlexer generator parameter count and forward-pass FLOPs.

Default setting targets the user request:
  - input_nc=3
  - output_nc=17
  - img_size=256
  - use_high_res=False (parity with existing 256px inference benchmarks)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
    p = argparse.ArgumentParser(description="Measure HistoPlexer params and FLOPs (forward only).")
    p.add_argument("--device", default="cpu", help="Device for the forward pass (default: cpu)")
    p.add_argument("--img-size", type=int, default=256, help="Input H=W (default: 256)")
    p.add_argument("--input-nc", type=int, default=3, help="Input channels (default: 3)")
    p.add_argument("--output-nc", type=int, default=17, help="Output channels (default: 17)")
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

    from src.models.generator import unet_translator  # noqa: E402

    dev = torch.device(str(args.device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model = unet_translator(
        input_nc=int(args.input_nc),
        output_nc=int(args.output_nc),
        use_high_res=bool(args.use_high_res),
        use_multiscale=bool(args.use_multiscale),
        ngf=int(args.ngf),
        depth=int(args.depth),
        encoder_padding=int(args.encoder_padding),
        decoder_padding=int(args.decoder_padding),
        extra_feature_size=int(args.extra_feature_size),
        device=dev,
    ).to(dev)
    model.eval()

    x = torch.zeros((1, int(args.input_nc), int(args.img_size), int(args.img_size)), device=dev, dtype=torch.float32)
    extra = None
    if int(args.extra_feature_size) > 0:
        extra = torch.zeros((1, int(args.extra_feature_size)), device=dev, dtype=torch.float32)

    class _ForwardOnly(torch.nn.Module):
        def __init__(self, base_model: torch.nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, x_in: torch.Tensor, extra_in: Optional[torch.Tensor]):
            return self.base_model(x_in, extra_in)[-1]

    forward_only = _ForwardOnly(model).to(dev)
    forward_only.eval()

    inputs: tuple[Any, ...] = (x, extra)

    with torch.inference_mode():
        for _ in range(max(0, int(args.warmup))):
            _ = forward_only(x, extra)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    params_total, params_trainable = _count_params(model)

    flops, flops_source = _estimate_flops(forward_only, inputs, backend=str(args.flops_backend), device=str(dev))

    row = {
        "method": "HistoPlexer",
        "device": str(dev),
        "img_size": int(args.img_size),
        "input_nc": int(args.input_nc),
        "output_nc": int(args.output_nc),
        "use_high_res": bool(args.use_high_res),
        "use_multiscale": bool(args.use_multiscale),
        "extra_feature_size": int(args.extra_feature_size),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "flops": ("" if flops is None else int(flops)),
        "flops_source": str(flops_source),
    }

    flops_str = "n/a" if flops is None else f"{float(flops) / 1e9:.3f} GFLOPs"
    print(
        f"[ok] HistoPlexer out_nc={row['output_nc']} in_nc={row['input_nc']} img={row['img_size']} "
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
                    "use_high_res",
                    "use_multiscale",
                    "extra_feature_size",
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
