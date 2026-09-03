#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HEX training peak CUDA memory (batch_size=1) across output dimensions C.

Assumptions (as requested):
  - input fixed RGB: 3 channels
  - sweep output dim C in [3, 7, 17, 28, 60] by default

This approximates one training step from `hex/train_dist_sp_fds_paper.py`:
  - CustomModelFDS forward
  - robust_loss_pytorch adaptive loss
  - AMP + GradScaler
  - optimizer = Adam(trainable params) + criterion_ad params group

Important:
  MUSK backbone weights must be present at ~/.cache/model.safetensors (or provided via --musk-safetensors).
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


def _set_trainable(model, stage: str) -> None:
    # Mirrors the schedule in hex/train_dist_sp_fds_paper.py (stage1/stage2),
    # plus a convenience 'all' mode.
    for p in model.parameters():
        p.requires_grad = False

    if stage == "stage1":
        for layer in model.visual.beit3.encoder.layers[-4:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in model.visual.beit3.encoder.layer_norm.parameters():
            p.requires_grad = True
    elif stage == "stage2":
        pass
    elif stage == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown train stage: {stage!r} (use stage1, stage2, all)")

    for p in model.regression_head.parameters():
        p.requires_grad = True
    for p in model.regression_head1.parameters():
        p.requires_grad = True


def _make_criterion_ad(*, c: int, device: str, cuda_idx: Optional[int]):
    import robust_loss_pytorch
    import torch

    if device.startswith("cuda"):
        # robust_loss_pytorch expects a device spec; training script passes local_rank (int).
        try:
            return robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                num_dims=c, float_dtype=torch.float32, device=int(cuda_idx if cuda_idx is not None else 0)
            )
        except Exception:
            return robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                num_dims=c, float_dtype=torch.float32, device=torch.device(device)
            )

    return robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=c, float_dtype=torch.float32, device="cpu")


def _measure_hex_train_peak(
    *,
    c: int,
    device: str,
    cuda_idx: Optional[int],
    img_size: int,
    musk_img_size: int,
    warmup_steps: int,
    precision: str,
    train_stage: str,
    lr: float,
) -> dict[str, Any]:
    import torch

    from hex.hex_architecture_fds import CustomModelFDS

    dev = torch.device(device)
    model = CustomModelFDS(visual_output_dim=1024, num_outputs=c, musk_img_size=musk_img_size).to(dev)
    _set_trainable(model, train_stage)

    model.train()
    model.training_status = True

    criterion_ad = _make_criterion_ad(c=c, device=device, cuda_idx=cuda_idx)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.add_param_group({"params": criterion_ad.parameters(), "lr": lr, "name": "criterion_ad"})

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and precision == "fp16"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and precision == "fp16"))

    dtype_in = torch.float16 if (device.startswith("cuda") and precision in {"fp16", "bf16"}) else torch.float32
    inputs = torch.zeros((1, 3, img_size, img_size), device=dev, dtype=dtype_in)
    labels = torch.zeros((1, c), device=dev, dtype=dtype_in)

    # FDS moments bookkeeping (small, but keeps parity with training loop).
    moments = model.FDS.init_moments(device=dev)

    def train_step() -> float:
        nonlocal moments
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, precision):
            outputs, raw_features = model(inputs, labels, epoch=0)
            diff = outputs.to(dtype=torch.float32) - labels.to(dtype=torch.float32)
            loss = torch.mean(criterion_ad.lossfun(diff))
        if device.startswith("cuda") and precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        moments = model.FDS.accumulate_moments(
            moments,
            raw_features.detach().to(dtype=torch.float32),
            labels.detach().to(dtype=torch.float32),
        )
        return float(loss.detach().item())

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(max(warmup_steps, 0)):
        _ = train_step()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    loss_value = train_step()

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        peak_alloc = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak_alloc = 0
        peak_reserved = 0

    del model, inputs, labels, optimizer, criterion_ad
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "baseline_allocated_mb": _mb(baseline_alloc),
        "baseline_reserved_mb": _mb(baseline_reserved),
        "peak_allocated_mb": _mb(peak_alloc),
        "peak_reserved_mb": _mb(peak_reserved),
        "loss": float(loss_value),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure HEX training peak CUDA memory (input=3, sweep C).")
    parser.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    parser.add_argument("--img-size", type=int, default=384, help="Dummy input size H=W (default: 384)")
    parser.add_argument("--musk-img-size", type=int, default=384, help="MUSK config img_size (default: 384)")
    parser.add_argument(
        "--channels",
        type=_split_ints,
        default="3,7,17,28,60",
        help="Comma-separated output dims C (default: 3,7,17,28,60)",
    )
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup train steps (default: 1)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Autocast precision (default: fp16)",
    )
    parser.add_argument(
        "--train-stage",
        choices=["stage1", "stage2", "all"],
        default="stage1",
        help="Trainable scope (default: stage1)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument(
        "--musk-safetensors",
        type=Path,
        default=None,
        help="Optional local model.safetensors to copy to ~/.cache/model.safetensors (avoids download).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("hex_train_peak_memory.csv"),
        help="Output CSV path (default: hex_train_peak_memory.csv)",
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
        result = _measure_hex_train_peak(
            c=c,
            device=device,
            cuda_idx=cuda_idx,
            img_size=int(args.img_size),
            musk_img_size=int(args.musk_img_size),
            warmup_steps=int(args.warmup_steps),
            precision=str(args.precision),
            train_stage=str(args.train_stage),
            lr=float(args.lr),
        )
        row = {
            "method": "HEX",
            "phase": "train",
            "device": device,
            "img_size": int(args.img_size),
            "musk_img_size": int(args.musk_img_size),
            "batch_size": 1,
            "input_nc": 3,
            "output_dim": c,
            "precision": str(args.precision),
            "train_stage": str(args.train_stage),
            "lr": float(args.lr),
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB "
            f"peak_reserved={row['peak_reserved_mb']:.2f}MB loss={row['loss']:.6f}"
        )

    print("\n=== Summary (HEX Training Peak Memory) ===")
    table_cols = ["C", "train_stage", "precision", "peak_allocated_mb", "peak_reserved_mb", "loss"]
    formatted = [
        [
            str(r["output_dim"]),
            str(r["train_stage"]),
            str(r["precision"]),
            f"{float(r['peak_allocated_mb']):.2f}",
            f"{float(r['peak_reserved_mb']):.2f}",
            f"{float(r['loss']):.6f}",
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
            "train_stage",
            "lr",
            "baseline_allocated_mb",
            "baseline_reserved_mb",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "loss",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"[ok] wrote CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

