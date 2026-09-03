#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure CUT training peak CUDA memory for batch_size=1 across channel counts.

Setting (as requested):
  - input_nc = output_nc = C
  - full CUT training step (G + D + F / NCE)
  - batch_size = 1
  - H = W = 256 (configurable)
  - sweep C in [3, 7, 17, 28, 60] by default

Implementation notes:
  - Uses CUT's own model + optimize_parameters() to reflect real training behavior.
  - Calls data_dependent_initialize(...) first to materialize netF (mlp_sample) and optimizer_F.
  - Runs one warmup optimize_parameters() to allocate Adam optimizer state, then measures one step.

Example:
  python benchmark/methods/CUT/measure_train_peak_memory.py --device cuda:0
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from options.train_options import TrainOptions  # noqa: E402
from models import create_model  # noqa: E402


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


def _make_train_opt(
    *,
    c: int,
    device: str,
    gpu_ids: list[int],
    img_size: int,
    checkpoints_dir: str,
    name: str,
    netG: str,
    netD: str,
    ngf: int,
    ndf: int,
    normG: str,
    normD: str,
    no_antialias: bool,
    no_antialias_up: bool,
    use_dropout: bool,
    cut_mode: str,
    lambda_gan: float,
    lambda_nce: float,
    nce_layers: str,
) -> Any:
    # TrainOptions.parse() writes to disk via print_options(); avoid that by calling gather_options().
    # We still use TrainOptions to ensure all CUT-specific flags exist.
    cmd_line = " ".join(
        [
            "--dataroot placeholder",
            f"--name {name}",
            f"--checkpoints_dir {checkpoints_dir}",
            "--model cut",
            "--dataset_mode unaligned",
            "--direction AtoB",
            "--preprocess none",
            f"--load_size {img_size}",
            f"--crop_size {img_size}",
            "--batch_size 1",
            f"--input_nc {c}",
            f"--output_nc {c}",
            f"--netG {netG}",
            f"--netD {netD}",
            f"--ngf {ngf}",
            f"--ndf {ndf}",
            f"--normG {normG}",
            f"--normD {normD}",
            f"--CUT_mode {cut_mode}",
            f"--lambda_GAN {lambda_gan}",
            f"--lambda_NCE {lambda_nce}",
            f"--nce_layers {nce_layers}",
            "--netF mlp_sample",
            "--netF_nc 256",
            "--nce_T 0.07",
            "--num_patches 256",
            "--flip_equivariance False",
            "--gan_mode lsgan",
            "--lr 0.0002",
            "--beta1 0.5",
            "--beta2 0.999",
            "--lr_policy linear",
            "--total_iterations 2",
            f"--gpu_ids {gpu_ids[0] if gpu_ids else -1}",
            f"--no_dropout {'False' if use_dropout else 'True'}",
        ]
        + (["--no_antialias"] if no_antialias else [])
        + (["--no_antialias_up"] if no_antialias_up else [])
    )

    options = TrainOptions(cmd_line=cmd_line)
    opt = options.gather_options()

    # Mirror BaseOptions.parse() logic (without print_options()).
    opt.isTrain = True

    str_ids = str(opt.gpu_ids).split(",")
    opt.gpu_ids = []
    for sid in str_ids:
        sid = sid.strip()
        if not sid:
            continue
        idx = int(sid)
        if idx >= 0:
            opt.gpu_ids.append(idx)

    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def _make_dummy_batch(*, device: str, c: int, img_size: int) -> dict[str, Any]:
    import torch

    if device == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    a = torch.zeros((1, c, img_size, img_size), device=dev, dtype=torch.float32)
    b = torch.zeros((1, c, img_size, img_size), device=dev, dtype=torch.float32)
    return {
        "A": a,
        "B": b,
        "A_paths": ["dummy_A"],
        "B_paths": ["dummy_B"],
    }


def _measure_cut_train_peak(
    *,
    device: str,
    c: int,
    img_size: int,
    checkpoints_dir: str,
    name: str,
    netG: str,
    netD: str,
    ngf: int,
    ndf: int,
    normG: str,
    normD: str,
    no_antialias: bool,
    no_antialias_up: bool,
    use_dropout: bool,
    cut_mode: str,
    lambda_gan: float,
    lambda_nce: float,
    nce_layers: str,
) -> dict[str, Any]:
    import torch

    device_str, gpu_ids = _parse_device(device)
    opt = _make_train_opt(
        c=c,
        device=device_str,
        gpu_ids=gpu_ids,
        img_size=img_size,
        checkpoints_dir=checkpoints_dir,
        name=name,
        netG=netG,
        netD=netD,
        ngf=ngf,
        ndf=ndf,
        normG=normG,
        normD=normD,
        no_antialias=no_antialias,
        no_antialias_up=no_antialias_up,
        use_dropout=use_dropout,
        cut_mode=cut_mode,
        lambda_gan=lambda_gan,
        lambda_nce=lambda_nce,
        nce_layers=nce_layers,
    )

    model = create_model(opt)

    batch = _make_dummy_batch(device=device_str, c=c, img_size=img_size)

    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Ensure netF / optimizer_F exist.
    model.data_dependent_initialize(batch)

    # Warmup step to allocate Adam optimizer state.
    model.set_input(batch)
    model.optimize_parameters()

    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    baseline_alloc = int(torch.cuda.memory_allocated()) if device_str.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved()) if device_str.startswith("cuda") else 0

    model.set_input(batch)
    model.optimize_parameters()

    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_alloc = 0
        peak_reserved = 0

    # Try to surface a representative scalar loss if available.
    losses = {}
    try:
        losses = model.get_current_losses()
    except Exception:
        losses = {}

    out = {
        "baseline_allocated_mb": _mb(baseline_alloc),
        "baseline_reserved_mb": _mb(baseline_reserved),
        "peak_allocated_mb": _mb(peak_alloc),
        "peak_reserved_mb": _mb(peak_reserved),
        "losses_json": str(losses),
    }

    del model
    del batch
    if device_str.startswith("cuda"):
        torch.cuda.empty_cache()

    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure CUT training peak CUDA memory (full CUT: G+D+F).")
    parser.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    parser.add_argument("--img-size", type=int, default=256, help="H=W (default: 256)")
    parser.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated channel counts C (default: 3,7,17,28,60)",
    )
    parser.add_argument("--netG", default="resnet_9blocks", help="Generator arch (default: resnet_9blocks)")
    parser.add_argument("--netD", default="basic", help="Discriminator arch (default: basic)")
    parser.add_argument("--ngf", type=int, default=64, help="Generator base channels (default: 64)")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator base channels (default: 64)")
    parser.add_argument("--normG", default="instance", help="Normalization for G (default: instance)")
    parser.add_argument("--normD", default="instance", help="Normalization for D (default: instance)")
    parser.add_argument("--use_dropout", action="store_true", help="Enable dropout in G (default: disabled)")
    parser.add_argument("--no_antialias", action="store_true", help="Use stride conv downsampling")
    parser.add_argument("--no_antialias_up", action="store_true", help="Use ConvTranspose upsampling")
    parser.add_argument("--CUT_mode", default="cut", help="CUT_mode (default: cut)")
    parser.add_argument("--lambda_GAN", type=float, default=1.0, help="GAN loss weight (default: 1.0)")
    parser.add_argument("--lambda_NCE", type=float, default=1.0, help="NCE loss weight (default: 1.0)")
    parser.add_argument("--nce_layers", default="0,4,8,12,16", help="NCE layers (default: 0,4,8,12,16)")
    parser.add_argument(
        "--checkpoints-dir",
        default="/tmp/cut_peak_mem",
        help="Dummy checkpoints_dir for CUT BaseModel (default: /tmp/cut_peak_mem)",
    )
    parser.add_argument(
        "--name",
        default="cut_peak_mem",
        help="Dummy experiment name for CUT BaseModel (default: cut_peak_mem)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("cut_train_peak_memory.csv"),
        help="Output CSV path (default: cut_train_peak_memory.csv)",
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for c in args.channels:
        c = int(c)
        result = _measure_cut_train_peak(
            device=str(args.device),
            c=c,
            img_size=int(args.img_size),
            checkpoints_dir=str(args.checkpoints_dir),
            name=str(args.name),
            netG=str(args.netG),
            netD=str(args.netD),
            ngf=int(args.ngf),
            ndf=int(args.ndf),
            normG=str(args.normG),
            normD=str(args.normD),
            no_antialias=bool(args.no_antialias),
            no_antialias_up=bool(args.no_antialias_up),
            use_dropout=bool(args.use_dropout),
            cut_mode=str(args.CUT_mode),
            lambda_gan=float(args.lambda_GAN),
            lambda_nce=float(args.lambda_NCE),
            nce_layers=str(args.nce_layers),
        )
        row = {
            "method": "CUT",
            "phase": "train",
            "device": str(args.device),
            "img_size": int(args.img_size),
            "batch_size": 1,
            "input_nc": c,
            "output_nc": c,
            "netG": str(args.netG),
            "netD": str(args.netD),
            "ngf": int(args.ngf),
            "ndf": int(args.ndf),
            "normG": str(args.normG),
            "normD": str(args.normD),
            "CUT_mode": str(args.CUT_mode),
            "lambda_GAN": float(args.lambda_GAN),
            "lambda_NCE": float(args.lambda_NCE),
            "nce_layers": str(args.nce_layers),
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB "
            f"peak_reserved={row['peak_reserved_mb']:.2f}MB losses={row['losses_json']}"
        )

    print("\n=== Summary (CUT Training Peak Memory, full CUT) ===")
    table_cols = ["C", "peak_allocated_mb", "peak_reserved_mb"]
    formatted = [
        [str(r["input_nc"]), f"{float(r['peak_allocated_mb']):.2f}", f"{float(r['peak_reserved_mb']):.2f}"]
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
            "batch_size",
            "input_nc",
            "output_nc",
            "netG",
            "netD",
            "ngf",
            "ndf",
            "normG",
            "normD",
            "CUT_mode",
            "lambda_GAN",
            "lambda_NCE",
            "nce_layers",
            "baseline_allocated_mb",
            "baseline_reserved_mb",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "losses_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"[ok] wrote CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
