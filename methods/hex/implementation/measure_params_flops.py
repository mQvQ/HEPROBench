#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HEX parameter count and forward-pass FLOPs.

Important:
  HEX's MUSK backbone may try to download weights if not cached.
  This script refuses to run unless `~/.cache/model.safetensors` exists,
  or you provide `--musk-safetensors` to copy into that location.

Default setting targets the user request:
  - input_nc=3 (fixed)
  - output_nc=17
  - img_size=384 (matches HEX sp_infer default)
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


_ROOT = Path(__file__).resolve().parent  # benchmark/methods/HEX
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
    parser = argparse.ArgumentParser(description="Measure HEX params and FLOPs (forward only).")
    parser.add_argument("--device", default="cpu", help="Device for the forward pass (default: cpu)")
    parser.add_argument("--img-size", type=int, default=384, help="Dummy input size H=W (default: 384)")
    parser.add_argument("--musk-img-size", type=int, default=384, help="MUSK config img_size (default: 384)")
    parser.add_argument("--output-nc", type=int, default=17, help="Output dimension C (default: 17)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards before FLOPs profile (default: 1)")
    parser.add_argument(
        "--flops-backend",
        choices=["auto", "fvcore", "profiler", "none"],
        default="auto",
        help="FLOPs backend preference (default: auto)",
    )
    parser.add_argument(
        "--musk-safetensors",
        type=Path,
        default=None,
        help="Optional local model.safetensors to copy to ~/.cache/model.safetensors (avoids download).",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args(argv)

    _require_torch()
    _ensure_musk_weights(args.musk_safetensors)
    import torch

    dev = torch.device(str(args.device))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    from hex.hex_architecture_fds import CustomModelFDS

    model = CustomModelFDS(visual_output_dim=1024, num_outputs=int(args.output_nc), musk_img_size=int(args.musk_img_size)).to(
        dev
    )
    model.eval()
    model.training_status = False

    x = torch.zeros((1, 3, int(args.img_size), int(args.img_size)), device=dev, dtype=torch.float32)

    class _ForwardOnly(torch.nn.Module):
        def __init__(self, base_model: torch.nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            out, _raw = self.base_model(x_in, None, epoch=0)
            return out

    forward_only = _ForwardOnly(model).to(dev)
    forward_only.eval()

    with torch.inference_mode():
        for _ in range(max(0, int(args.warmup))):
            _ = forward_only(x)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    params_total, params_trainable = _count_params(model)
    flops, flops_source = _estimate_flops(forward_only, (x,), backend=str(args.flops_backend), device=str(dev))

    row = {
        "method": "HEX",
        "device": str(dev),
        "img_size": int(args.img_size),
        "input_nc": 3,
        "output_nc": int(args.output_nc),
        "musk_img_size": int(args.musk_img_size),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "flops": ("" if flops is None else int(flops)),
        "flops_source": str(flops_source),
    }

    flops_str = "n/a" if flops is None else f"{float(flops) / 1e9:.3f} GFLOPs"
    print(
        f"[ok] HEX out_nc={row['output_nc']} img={row['img_size']} musk_img={row['musk_img_size']} "
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
                    "musk_img_size",
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
