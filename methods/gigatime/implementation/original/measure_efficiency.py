#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import time
from pathlib import Path


def _autocast(device: str, precision: str):
    import torch

    if not device.startswith("cuda") or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the original binary-segmentation GigaTIME model")
    parser.add_argument("--mode", choices=["params", "latency"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--img-size", type=int, default=556)
    parser.add_argument("--output-nc", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from scripts.archs import gigatime

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = gigatime(num_classes=args.output_nc, input_channels=3, sigmoid=True).to(device).eval()
    params_total = sum(int(value.numel()) for value in model.parameters())
    params_trainable = sum(int(value.numel()) for value in model.parameters() if value.requires_grad)
    x = torch.zeros((args.batch_size, 3, args.img_size, args.img_size), dtype=torch.float32)
    row = {
        "method": "gigatime_original",
        "device": str(device),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "nc_in": 3,
        "nc_out": args.output_nc,
        "precision": args.precision,
        "params_total": params_total,
        "params_trainable": params_trainable,
    }
    if args.mode == "params":
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        x_device = x.to(device)
        with torch.inference_mode(), _autocast(args.device, args.precision):
            for _ in range(args.warmup):
                model(x_device)
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            with torch.inference_mode(), _autocast(args.device, args.precision):
                model(x_device)
        row["flops"] = sum(int(event.flops or 0) for event in profiler.key_averages())
        row["flops_source"] = "torch.profiler"
    else:
        def run_once() -> None:
            output = model(x.to(device))
            output.detach().to("cpu")
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for _ in range(args.warmup):
            run_once()
        values = []
        with torch.inference_mode(), _autocast(args.device, args.precision):
            for _ in range(max(1, args.iters)):
                start = time.perf_counter()
                run_once()
                values.append((time.perf_counter() - start) * 1000.0)
        row.update(
            {
                "warmup": args.warmup,
                "iters": args.iters,
                "p50_ms_batch": float(np.percentile(values, 50)),
                "p90_ms_batch": float(np.percentile(values, 90)),
                "p99_ms_batch": float(np.percentile(values, 99)),
                "mean_ms_batch": sum(values) / len(values),
                "mean_ms_patch": sum(values) / (len(values) * args.batch_size),
                "total_ms": sum(values),
            }
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
