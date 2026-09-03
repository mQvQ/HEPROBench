#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure Pix2Pix/CycleGAN training peak CUDA memory (batch_size=1) across output dims C.

Assumptions (as requested):
  - input fixed RGB: 3 channels
  - sweep output dim C in [3, 7, 17, 28, 60] by default

Notes:
  - Runs one warmup step to allocate optimizer state, then measures one training step.
  - Avoids importing model classes (pix2pix_model imports pyvips); uses networks + loss code directly.
  - Avoids DataParallel (does not pass gpu_ids into define_G/define_D) for a cleaner single-GPU peak.
  - For cycle_gan with input_nc!=output_nc, lambda_identity must be 0 (default here).
"""

import argparse
import csv
import math
import sys
from itertools import chain
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


def _defaults_for(model: str) -> dict[str, Any]:
    m = model.strip().lower()
    if m == "pix2pix":
        return {
            "netG": "unet_256",
            "norm": "batch",
            "gan_mode": "vanilla",
            "use_dropout": True,  # opt.no_dropout default False in repo
        }
    if m in {"cycle_gan", "cyclegan"}:
        return {
            "netG": "resnet_9blocks",
            "norm": "instance",
            "gan_mode": "lsgan",
            "use_dropout": False,  # CycleGAN defaults no_dropout=True
        }
    raise ValueError(f"Unsupported model: {model!r} (use pix2pix or cycle_gan)")


def _set_requires_grad(model, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def _pix2pix_step(
    *,
    netG,
    netD,
    optimizer_G,
    optimizer_D,
    criterionGAN,
    criterionL1,
    real_A,
    real_B,
    lambda_L1: float,
) -> dict[str, Any]:
    import torch

    # forward
    fake_B = netG(real_A)

    # update D
    _set_requires_grad(netD, True)
    optimizer_D.zero_grad(set_to_none=True)
    fake_AB = torch.cat((real_A, fake_B), dim=1)
    pred_fake = netD(fake_AB.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    real_AB = torch.cat((real_A, real_B), dim=1)
    pred_real = netD(real_AB)
    loss_D_real = criterionGAN(pred_real, True)
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()
    optimizer_D.step()

    # update G
    _set_requires_grad(netD, False)
    optimizer_G.zero_grad(set_to_none=True)
    pred_fake_for_g = netD(fake_AB)
    loss_G_GAN = criterionGAN(pred_fake_for_g, True)
    loss_G_L1 = criterionL1(fake_B, real_B) * float(lambda_L1)
    loss_G = loss_G_GAN + loss_G_L1
    loss_G.backward()
    optimizer_G.step()

    return {
        "loss_D": float(loss_D.detach().item()),
        "loss_G": float(loss_G.detach().item()),
        "loss_G_L1": float(loss_G_L1.detach().item()),
        "out_shape": str(tuple(int(s) for s in fake_B.shape)),
        "out_dtype": str(fake_B.dtype),
    }


def _cycle_gan_step(
    *,
    netG_A,
    netG_B,
    netD_A,
    netD_B,
    optimizer_G,
    optimizer_D,
    fake_A_pool,
    fake_B_pool,
    criterionGAN,
    criterionCycle,
    real_A,
    real_B,
    lambda_A: float,
    lambda_B: float,
    lambda_identity: float,
) -> dict[str, Any]:
    import torch

    if float(lambda_identity) != 0.0:
        if real_A.shape[1] != real_B.shape[1]:
            raise ValueError("cycle_gan: lambda_identity must be 0 when input_nc != output_nc")

    # forward
    fake_B = netG_A(real_A)
    rec_A = netG_B(fake_B)
    fake_A = netG_B(real_B)
    rec_B = netG_A(fake_A)

    # G_A and G_B
    _set_requires_grad(netD_A, False)
    _set_requires_grad(netD_B, False)
    optimizer_G.zero_grad(set_to_none=True)

    loss_G_A = criterionGAN(netD_A(fake_B), True)
    loss_G_B = criterionGAN(netD_B(fake_A), True)
    loss_cycle_A = criterionCycle(rec_A, real_A) * float(lambda_A)
    loss_cycle_B = criterionCycle(rec_B, real_B) * float(lambda_B)

    if float(lambda_identity) > 0:
        idt_A = netG_B(real_A)
        idt_B = netG_A(real_B)
        loss_idt_A = criterionCycle(idt_A, real_A) * float(lambda_A) * float(lambda_identity)
        loss_idt_B = criterionCycle(idt_B, real_B) * float(lambda_B) * float(lambda_identity)
    else:
        loss_idt_A = torch.tensor(0.0, device=real_A.device)
        loss_idt_B = torch.tensor(0.0, device=real_A.device)

    loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
    loss_G.backward()
    optimizer_G.step()

    # D_A and D_B
    _set_requires_grad(netD_A, True)
    _set_requires_grad(netD_B, True)
    optimizer_D.zero_grad(set_to_none=True)

    # D_A
    fake_B_q = fake_B_pool.query(fake_B)
    pred_real = netD_A(real_B)
    loss_D_A_real = criterionGAN(pred_real, True)
    pred_fake = netD_A(fake_B_q.detach())
    loss_D_A_fake = criterionGAN(pred_fake, False)
    loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
    loss_D_A.backward()

    # D_B
    fake_A_q = fake_A_pool.query(fake_A)
    pred_real = netD_B(real_A)
    loss_D_B_real = criterionGAN(pred_real, True)
    pred_fake = netD_B(fake_A_q.detach())
    loss_D_B_fake = criterionGAN(pred_fake, False)
    loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
    loss_D_B.backward()

    optimizer_D.step()

    return {
        "loss_D": float((loss_D_A + loss_D_B).detach().item()),
        "loss_G": float(loss_G.detach().item()),
        "loss_cycle_A": float(loss_cycle_A.detach().item()),
        "loss_cycle_B": float(loss_cycle_B.detach().item()),
        "out_shape": str(tuple(int(s) for s in fake_B.shape)),
        "out_dtype": str(fake_B.dtype),
    }


def _measure_train_peak(
    *,
    model: str,
    c: int,
    device: str,
    cuda_idx: Optional[int],
    img_size: int,
    ngf: int,
    ndf: int,
    netG: str,
    netD: str,
    n_layers_D: int,
    norm: str,
    use_dropout: bool,
    gan_mode: str,
    init_type: str,
    init_gain: float,
    lr: float,
    beta1: float,
    warmup_steps: int,
    # pix2pix
    lambda_L1: float,
    # cyclegan
    lambda_A: float,
    lambda_B: float,
    lambda_identity: float,
    pool_size: int,
) -> dict[str, Any]:
    import torch

    from models import networks
    from util.image_pool import ImagePool

    dev = torch.device(device)
    if device.startswith("cuda"):
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)

    # dummy inputs
    real_A = torch.zeros((1, 3, img_size, img_size), device=dev, dtype=torch.float32)
    real_B = torch.zeros((1, c, img_size, img_size), device=dev, dtype=torch.float32)

    criterionGAN = networks.GANLoss(gan_mode).to(dev)
    criterionL1 = torch.nn.L1Loss()
    criterionCycle = torch.nn.L1Loss()

    if model == "pix2pix":
        netG_m = networks.define_G(
            3, c, ngf, netG, norm=norm, use_dropout=use_dropout, init_type=init_type, init_gain=init_gain, gpu_ids=[]
        ).to(dev)
        netD_m = networks.define_D(
            3 + c, ndf, netD, n_layers_D=n_layers_D, norm=norm, init_type=init_type, init_gain=init_gain, gpu_ids=[]
        ).to(dev)

        optimizer_G = torch.optim.Adam(netG_m.parameters(), lr=float(lr), betas=(float(beta1), 0.999))
        optimizer_D = torch.optim.Adam(netD_m.parameters(), lr=float(lr), betas=(float(beta1), 0.999))

        def step() -> dict[str, Any]:
            return _pix2pix_step(
                netG=netG_m,
                netD=netD_m,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                criterionGAN=criterionGAN,
                criterionL1=criterionL1,
                real_A=real_A,
                real_B=real_B,
                lambda_L1=lambda_L1,
            )

        models_to_del = [netG_m, netD_m, optimizer_G, optimizer_D]

    elif model == "cycle_gan":
        netG_A = networks.define_G(
            3, c, ngf, netG, norm=norm, use_dropout=use_dropout, init_type=init_type, init_gain=init_gain, gpu_ids=[]
        ).to(dev)
        netG_B = networks.define_G(
            c, 3, ngf, netG, norm=norm, use_dropout=use_dropout, init_type=init_type, init_gain=init_gain, gpu_ids=[]
        ).to(dev)
        netD_A = networks.define_D(
            c, ndf, netD, n_layers_D=n_layers_D, norm=norm, init_type=init_type, init_gain=init_gain, gpu_ids=[]
        ).to(dev)
        netD_B = networks.define_D(
            3, ndf, netD, n_layers_D=n_layers_D, norm=norm, init_type=init_type, init_gain=init_gain, gpu_ids=[]
        ).to(dev)

        optimizer_G = torch.optim.Adam(chain(netG_A.parameters(), netG_B.parameters()), lr=float(lr), betas=(float(beta1), 0.999))
        optimizer_D = torch.optim.Adam(chain(netD_A.parameters(), netD_B.parameters()), lr=float(lr), betas=(float(beta1), 0.999))

        fake_A_pool = ImagePool(int(pool_size))
        fake_B_pool = ImagePool(int(pool_size))

        def step() -> dict[str, Any]:
            return _cycle_gan_step(
                netG_A=netG_A,
                netG_B=netG_B,
                netD_A=netD_A,
                netD_B=netD_B,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                fake_A_pool=fake_A_pool,
                fake_B_pool=fake_B_pool,
                criterionGAN=criterionGAN,
                criterionCycle=criterionCycle,
                real_A=real_A,
                real_B=real_B,
                lambda_A=lambda_A,
                lambda_B=lambda_B,
                lambda_identity=lambda_identity,
            )

        models_to_del = [netG_A, netG_B, netD_A, netD_B, optimizer_G, optimizer_D]

    else:
        raise ValueError(f"Unsupported model: {model!r}")

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(max(int(warmup_steps), 0)):
        _ = step()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    result = step()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        peak_alloc = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak_alloc = 0
        peak_reserved = 0

    del real_A, real_B
    for obj in models_to_del:
        del obj
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "baseline_allocated_mb": _mb(baseline_alloc),
        "baseline_reserved_mb": _mb(baseline_reserved),
        "peak_allocated_mb": _mb(peak_alloc),
        "peak_reserved_mb": _mb(peak_reserved),
        **result,
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
    p = argparse.ArgumentParser(description="Measure Pix2Pix/CycleGAN training peak CUDA memory (input=3, sweep C).")
    p.add_argument("--model", choices=["pix2pix", "cycle_gan"], default="pix2pix", help="Model family (default: pix2pix)")
    p.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    p.add_argument("--channels", type=_split_ints, default="3,7,17,28,60", help="Comma-separated C (default: 3,7,17,28,60)")
    p.add_argument("--img-size", type=int, default=256, help="Input H=W (default: 256)")
    p.add_argument("--warmup-steps", type=int, default=1, help="Warmup train steps before measuring (default: 1)")

    p.add_argument("--ngf", type=int, default=64, help="Generator base channels (default: 64)")
    p.add_argument("--ndf", type=int, default=64, help="Discriminator base channels (default: 64)")
    p.add_argument("--netG", type=str, default=None, help="Generator arch (default depends on --model)")
    p.add_argument("--netD", type=str, default="basic", help="Discriminator arch (default: basic)")
    p.add_argument("--n_layers_D", type=int, default=3, help="n_layers_D if netD==n_layers (default: 3)")
    p.add_argument("--norm", type=str, default=None, choices=["batch", "instance", "none"], help="Norm (default depends on --model)")
    p.add_argument("--use-dropout", action="store_true", help="Enable dropout in G (overrides model default)")
    p.add_argument("--no-dropout", action="store_true", help="Disable dropout in G (overrides model default)")
    p.add_argument("--gan-mode", type=str, default=None, choices=["vanilla", "lsgan", "wgangp"], help="GAN objective (default depends on --model)")
    p.add_argument("--init-type", type=str, default="normal", help="Init type (default: normal)")
    p.add_argument("--init-gain", type=float, default=0.02, help="Init gain (default: 0.02)")

    p.add_argument("--lr", type=float, default=0.0002, help="Adam LR (default: 2e-4)")
    p.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (default: 0.5)")

    # pix2pix
    p.add_argument("--lambda-L1", type=float, default=100.0, help="Pix2Pix L1 weight (default: 100.0)")

    # cyclegan
    p.add_argument("--lambda-A", type=float, default=10.0, help="CycleGAN cycle weight A (default: 10.0)")
    p.add_argument("--lambda-B", type=float, default=10.0, help="CycleGAN cycle weight B (default: 10.0)")
    p.add_argument(
        "--lambda-identity",
        type=float,
        default=0.0,
        help="CycleGAN identity weight (default: 0.0; must be 0 when input_nc!=output_nc)",
    )
    p.add_argument(
        "--pool-size",
        type=int,
        default=0,
        help="CycleGAN fake image pool size (default: 0; set 50 to match repo defaults but uses extra GPU memory)",
    )

    p.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: <model>_train_peak_memory.csv)")

    args = p.parse_args(argv)

    model = str(args.model)
    defaults = _defaults_for(model)
    netG = str(args.netG) if args.netG is not None else str(defaults["netG"])
    norm = str(args.norm) if args.norm is not None else str(defaults["norm"])
    gan_mode = str(args.gan_mode) if args.gan_mode is not None else str(defaults["gan_mode"])

    if bool(args.use_dropout) and bool(args.no_dropout):
        raise ValueError("Cannot set both --use-dropout and --no-dropout")
    if bool(args.use_dropout):
        use_dropout = True
    elif bool(args.no_dropout):
        use_dropout = False
    else:
        use_dropout = bool(defaults["use_dropout"])

    device, cuda_idx = _parse_device(str(args.device))
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = _ROOT / f"{model}_train_peak_memory.csv"

    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)

    rows: list[dict[str, Any]] = []
    for c in args.channels:
        c = int(c)
        result = _measure_train_peak(
            model=model,
            c=c,
            device=device,
            cuda_idx=cuda_idx,
            img_size=int(args.img_size),
            ngf=int(args.ngf),
            ndf=int(args.ndf),
            netG=netG,
            netD=str(args.netD),
            n_layers_D=int(args.n_layers_D),
            norm=norm,
            use_dropout=use_dropout,
            gan_mode=gan_mode,
            init_type=str(args.init_type),
            init_gain=float(args.init_gain),
            lr=float(args.lr),
            beta1=float(args.beta1),
            warmup_steps=int(args.warmup_steps),
            lambda_L1=float(args.lambda_L1),
            lambda_A=float(args.lambda_A),
            lambda_B=float(args.lambda_B),
            lambda_identity=float(args.lambda_identity),
            pool_size=int(args.pool_size),
        )

        row = {
            "method": "pytorch-CycleGAN-and-pix2pix",
            "model": model,
            "phase": "train",
            "device": device,
            "batch_size": 1,
            "input_nc": 3,
            "output_nc": c,
            "img_size": int(args.img_size),
            "ngf": int(args.ngf),
            "ndf": int(args.ndf),
            "netG": netG,
            "netD": str(args.netD),
            "n_layers_D": int(args.n_layers_D),
            "norm": norm,
            "use_dropout": bool(use_dropout),
            "gan_mode": gan_mode,
            "lr": float(args.lr),
            "beta1": float(args.beta1),
            "lambda_L1": float(args.lambda_L1),
            "lambda_A": float(args.lambda_A),
            "lambda_B": float(args.lambda_B),
            "lambda_identity": float(args.lambda_identity),
            "pool_size": int(args.pool_size),
            **result,
        }
        rows.append(row)
        print(
            f"[ok] {model} C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB peak_reserved={row['peak_reserved_mb']:.2f}MB "
            f"lossD={_safe_float(row.get('loss_D')) or 0.0:.4f} lossG={_safe_float(row.get('loss_G')) or 0.0:.4f}"
        )

    print(f"\n=== Summary ({model} Training Peak Memory) ===")
    table_cols = ["C", "peak_allocated_mb", "peak_reserved_mb", "loss_D", "loss_G"]
    formatted = [
        [
            str(r["output_nc"]),
            f"{float(r['peak_allocated_mb']):.2f}",
            f"{float(r['peak_reserved_mb']):.2f}",
            f"{_safe_float(r.get('loss_D')) or 0.0:.4f}",
            f"{_safe_float(r.get('loss_G')) or 0.0:.4f}",
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
            "ndf",
            "netG",
            "netD",
            "n_layers_D",
            "norm",
            "use_dropout",
            "gan_mode",
            "lr",
            "beta1",
            "lambda_L1",
            "lambda_A",
            "lambda_B",
            "lambda_identity",
            "pool_size",
            "baseline_allocated_mb",
            "baseline_reserved_mb",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "loss_D",
            "loss_G",
            "loss_G_L1",
            "loss_cycle_A",
            "loss_cycle_B",
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

