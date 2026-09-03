#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Measure HistoPlexer training peak CUDA memory (batch_size=1) across output dims C.

Assumptions (as requested):
  - input fixed RGB: 3 channels
  - sweep output dim C in [3, 7, 17, 28, 60] by default

This script approximates a single training iteration (D step + G step) from
`src/trainers/histoplexer_trainer.py`, including:
  - LS-GAN losses
  - optional multiscale + high-res logic
  - optional R1 regularization (worst-case by measuring at step=0)
  - optional ASP (PatchNCE-style) loss components when w_ASP > 0
  - optional GaussPyramidLoss (requires kornia) when use_gp=True

Notes:
  - To avoid any implicit network downloads, this script never requests pretrained weights.
    If feature encoder is enabled (use_feat_enc=True), it uses a randomly initialized VGG19
    unless you pass --vgg-path to load a local checkpoint.
"""

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
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


def _center_crop(x, size: int):
    import torchvision.transforms.functional as ttf

    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return ttf.center_crop(x, [size, size])


def _resize_tensor(tensor, output_sizes: list[int]):
    import torchvision.transforms.functional as ttf

    out = []
    for osz in output_sizes:
        out.append(ttf.resize(tensor, osz))
    return out


def _add_noise_prob(tensor, *, factor: float = 0.1, p: float = 0.5):
    import random
    import torch

    if random.random() < p:
        tensor_noise = factor * torch.rand(tensor.size(), device=tensor.device, dtype=tensor.dtype)
        return tensor + tensor_noise
    return tensor


def _get_r1(*, D, src, tgt_list, gamma_0: float, lazy_c: float):
    import torch

    total_r1_loss = 0.0
    # R1 is typically computed in fp32 (and avoids dtype mismatches if the main step runs with autocast).
    src = src.detach().to(dtype=torch.float32).requires_grad_(True)
    tgt_list = [tgt.detach().to(dtype=torch.float32).requires_grad_(True) for tgt in tgt_list]

    score_maps = D((src, tgt_list))
    for i, score_map in enumerate(score_maps):
        tgt = tgt_list[len(tgt_list) - i - 1]
        bs = tgt.shape[0]
        img_size = tgt.shape[-1]
        r1_gamma = float(gamma_0) * float(img_size**2 / bs)

        (r1_grad,) = torch.autograd.grad(
            outputs=[score_map.sum()],
            inputs=[tgt],
            create_graph=True,
            only_inputs=True,
        )
        r1_penalty = r1_grad.square().sum(dim=[1, 2, 3])
        r1_loss = r1_penalty.mean() * (r1_gamma / 2.0) * float(lazy_c)
        total_r1_loss = total_r1_loss + r1_loss

    return total_r1_loss


def _build_vgg19_feature_slices(*, device, vgg_path: Optional[Path]):
    import torch
    import torch.nn as nn
    import torchvision

    class _Normalization(nn.Module):
        def __init__(self, device_):
            super().__init__()
            mean = torch.tensor([0.485, 0.456, 0.406], device=device_)
            std = torch.tensor([0.229, 0.224, 0.225], device=device_)
            self.register_buffer("mean", mean.view(-1, 1, 1))
            self.register_buffer("std", std.view(-1, 1, 1))

        def forward(self, img):
            return (img - self.mean) / self.std

    class _VGG19(nn.Module):
        def __init__(self):
            super().__init__()
            vgg19 = torchvision.models.vgg19(weights=None)
            if vgg_path is not None:
                sd = torch.load(str(vgg_path), map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                try:
                    vgg19.load_state_dict(sd, strict=False)
                except Exception:
                    # For memory measurement, random weights are acceptable.
                    pass

            feats = vgg19.features
            self.slice1 = nn.Sequential(*[feats[i] for i in range(0, 2)])
            self.slice2 = nn.Sequential(*[feats[i] for i in range(2, 7)])
            self.slice3 = nn.Sequential(*[feats[i] for i in range(7, 12)])
            self.slice4 = nn.Sequential(*[feats[i] for i in range(12, 21)])
            self.slice5 = nn.Sequential(*[feats[i] for i in range(21, 30)])
            self.norm = _Normalization(device)

        def forward(self, x):
            if x.shape[1] == 1:
                x = torch.cat([x, x, x], dim=1)
            x = self.norm(x)
            h1 = self.slice1(x)
            h2 = self.slice2(h1)
            h3 = self.slice3(h2)
            h4 = self.slice4(h3)
            h5 = self.slice5(h4)
            return [h1, h2, h3, h4, h5]

    model = _VGG19().to(device)
    model.eval()
    return model


def _measure_train_peak(
    *,
    c: int,
    device: str,
    cuda_idx: Optional[int],
    patch_size: int,
    input_size: int,
    precision: str,
    warmup_steps: int,
    use_high_res: bool,
    use_multiscale: bool,
    ngf: int,
    depth: int,
    encoder_padding: int,
    decoder_padding: int,
    fm_feature_size: int,
    lr_G: float,
    lr_D: float,
    lr_F: float,
    beta_0: float,
    beta_1: float,
    use_gp: bool,
    w_L1: float,
    w_GP: float,
    w_ASP: float,
    w_R1: float,
    r1_gamma: float,
    r1_interval: int,
    ema_warmup: int,
    p_dis_add_noise: Optional[float],
    blur_gt: bool,
    use_feat_enc: bool,
    vgg_path: Optional[Path],
) -> dict[str, Any]:
    import torch
    import torchvision

    from src.models.discriminator import Discriminator
    from src.models.generator import unet_translator
    from src.models.patch_sampler import PatchSampleF
    from src.utils.loss.nce_loss import PatchNCELoss

    dev = torch.device(device)

    if device.startswith("cuda"):
        assert cuda_idx is not None
        torch.cuda.set_device(cuda_idx)

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
    d = Discriminator(
        input_nc=3,
        output_nc=c,
        use_high_res=use_high_res,
        use_multiscale=use_multiscale,
        ngf=ngf,
        depth=depth,
        device=dev,
    )
    g_ema = deepcopy(g).to(dev)
    g_ema.requires_grad_(False)

    opt_G = torch.optim.Adam(g.parameters(), lr=float(lr_G), betas=(float(beta_0), float(beta_1)))

    if float(w_R1) > 0:
        lazy_c = float(r1_interval) / float(r1_interval + 1) if int(r1_interval) > 1 else 1.0
        opt_D = torch.optim.Adam(
            d.parameters(),
            lr=float(lr_D) * lazy_c,
            betas=(float(beta_0) ** lazy_c, float(beta_1) ** lazy_c),
        )
    else:
        lazy_c = 1.0
        opt_D = torch.optim.Adam(d.parameters(), lr=float(lr_D), betas=(float(beta_0), float(beta_1)))

    if blur_gt:
        spatial_denoise = torchvision.transforms.GaussianBlur(3, sigma=1)
    else:
        spatial_denoise = None

    if use_multiscale:
        imc_sizes = [int(patch_size // (2 ** (2 * (j + 1)))) for j in reversed(range(int(depth // 2 - 1)))]
    else:
        imc_sizes = []

    if use_gp:
        try:
            from src.utils.loss.gp_loss import GaussPyramidLoss

            gp_loss = GaussPyramidLoss(channels=c)
        except Exception:
            gp_loss = None
    else:
        gp_loss = None

    l1_loss = torch.nn.L1Loss()

    if float(w_ASP) > 0:
        nce_loss = PatchNCELoss(batch_size=1, total_step=100000, n_step_decay=10000)
        f_model = PatchSampleF(device=dev)

        if use_feat_enc:
            e_model = _build_vgg19_feature_slices(device=dev, vgg_path=vgg_path)
            dummy = torch.randn((1, 1, patch_size, patch_size), device=dev, dtype=torch.float32)
            with torch.no_grad():
                dummy_feats = e_model(dummy)
        else:
            e_model = None
            dummy = torch.randn((1, 3, patch_size, patch_size), device=dev, dtype=torch.float32)
            with torch.no_grad():
                dummy_feats = g(dummy, extra_features=None, encode_only=True)

        f_model.init_model(dummy_input=dummy_feats)
        opt_F = torch.optim.Adam(f_model.parameters(), lr=float(lr_F), betas=(float(beta_0), float(beta_1)))
    else:
        nce_loss = None
        f_model = None
        e_model = None
        opt_F = None

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and precision == "fp16"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and precision == "fp16"))
    use_scaler = bool(device.startswith("cuda") and precision == "fp16")

    if precision == "fp16":
        dtype_in = torch.float16
    elif precision == "bf16":
        dtype_in = torch.bfloat16
    else:
        dtype_in = torch.float32

    he = torch.zeros((1, 3, input_size, input_size), device=dev, dtype=dtype_in)
    imc = torch.zeros((1, c, patch_size, patch_size), device=dev, dtype=dtype_in)
    feats_uni = None
    if fm_feature_size > 0:
        feats_uni = torch.zeros((1, fm_feature_size), device=dev, dtype=torch.float32)

    def _asp_loss(step: int, *, real_imc, fake_imc) -> torch.Tensor:
        assert float(w_ASP) > 0 and f_model is not None and nce_loss is not None
        fake_feats_per_ch = []
        real_feats_per_ch = []

        if use_feat_enc:
            assert e_model is not None
            with torch.no_grad():
                for i in range(real_imc.shape[1]):
                    fake_in = fake_imc[:, i : i + 1, :, :].repeat(1, 3, 1, 1).to(dtype=torch.float32)
                    real_in = real_imc[:, i : i + 1, :, :].repeat(1, 3, 1, 1).to(dtype=torch.float32)
                    fake_feats_per_ch.append(e_model(fake_in))
                    real_feats_per_ch.append(e_model(real_in))
        else:
            was_training = g.training
            g.eval()
            with torch.no_grad():
                for i in range(real_imc.shape[1]):
                    fake_in = fake_imc[:, i : i + 1, :, :].repeat(1, 3, 1, 1).to(dtype=torch.float32)
                    real_in = real_imc[:, i : i + 1, :, :].repeat(1, 3, 1, 1).to(dtype=torch.float32)
                    fake_feats_per_ch.append(g(fake_in, extra_features=None, encode_only=True))
                    real_feats_per_ch.append(g(real_in, extra_features=None, encode_only=True))
            if was_training:
                g.train()

        total_asp = []
        for feat_k, feat_q in zip(real_feats_per_ch, fake_feats_per_ch):
            total_asp_per_channel = 0.0
            n_layers = len(feat_q)

            feat_k_pool, sample_ids = f_model(feat_k, 256, None)
            feat_q_pool, _ = f_model(feat_q, 256, sample_ids)
            for f_q, f_k in zip(feat_q_pool, feat_k_pool):
                loss = nce_loss(f_q, f_k, step)
                total_asp_per_channel = total_asp_per_channel + loss.mean()

            total_asp_per_channel = total_asp_per_channel / float(n_layers)
            total_asp.append(total_asp_per_channel)

        total_asp = torch.stack(total_asp).mean()
        return total_asp

    def train_one_step(step: int) -> dict[str, Any]:
        g.train()
        d.train()
        if f_model is not None:
            f_model.train()

        # ---------------- D step ----------------
        for p in g.parameters():
            p.requires_grad_(False)
        for p in d.parameters():
            p.requires_grad_(True)

        opt_D.zero_grad(set_to_none=True)

        real_imcs = []
        tgt = imc
        if spatial_denoise is not None:
            tgt = spatial_denoise(tgt)
        if use_multiscale and imc_sizes:
            real_imcs.extend(_resize_tensor(tgt, imc_sizes))
        real_imcs.append(tgt)

        with torch.no_grad():
            with _autocast_ctx(device, precision):
                fake_imcs = g(he, feats_uni)

        expected_sizes = imc_sizes + [patch_size] if use_multiscale else [patch_size]
        fake_imcs = [_center_crop(x, expected_sizes[i]) for i, x in enumerate(fake_imcs)]
        real_imcs = [_center_crop(x, expected_sizes[i]) for i, x in enumerate(real_imcs)]

        if p_dis_add_noise is not None:
            p_val = float(p_dis_add_noise)
            fake_imcs = [_add_noise_prob(x, p=p_val) for x in fake_imcs]
            real_imcs = [_add_noise_prob(x, p=p_val) for x in real_imcs]

        with _autocast_ctx(device, precision):
            fake_score_maps = d((he, fake_imcs))
            real_score_maps = d((he, real_imcs))
            real_score_means = [m.mean(dim=(1, 2, 3)) for m in real_score_maps]
            fake_score_means = [m.mean(dim=(1, 2, 3)) for m in fake_score_maps]
            real_score_mean = sum(real_score_means) / len(real_score_means)
            fake_score_mean = sum(fake_score_means) / len(fake_score_means)
            real_label = torch.ones_like(real_score_mean, device=dev)
            loss_D = 0.5 * (real_score_mean - real_label).square().mean() + 0.5 * (fake_score_mean).square().mean()

        if float(w_R1) > 0 and (int(step) % int(r1_interval) == 0):
            loss_R1 = float(w_R1) * _get_r1(D=d, src=he, tgt_list=real_imcs, gamma_0=float(r1_gamma), lazy_c=float(lazy_c))
        else:
            loss_R1 = torch.tensor(0.0, device=dev)

        loss_D_total = loss_D + loss_R1
        if use_scaler and scaler is not None:
            scaler.scale(loss_D_total).backward()
            scaler.step(opt_D)
        else:
            loss_D_total.backward()
            opt_D.step()

        # ---------------- G step ----------------
        for p in g.parameters():
            p.requires_grad_(True)
        for p in d.parameters():
            p.requires_grad_(False)

        opt_G.zero_grad(set_to_none=True)
        if opt_F is not None:
            opt_F.zero_grad(set_to_none=True)

        with _autocast_ctx(device, precision):
            fake_imcs_g = g(he, feats_uni)
        fake_imcs_g = [_center_crop(x, expected_sizes[i]) for i, x in enumerate(fake_imcs_g)]
        fake_imc_final = fake_imcs_g[-1]

        if gp_loss is not None and float(w_GP) > 0:
            loss_L1 = float(w_GP) * gp_loss(fake_imc_final.float(), imc.float())
        elif use_gp and float(w_GP) > 0 and gp_loss is None:
            loss_L1 = float(w_GP) * l1_loss(fake_imc_final.float(), imc.float())
        else:
            loss_L1 = float(w_L1) * l1_loss(fake_imc_final.float(), imc.float())

        if float(w_ASP) > 0:
            loss_ASP = float(w_ASP) * _asp_loss(int(step), real_imc=imc, fake_imc=fake_imc_final)
        else:
            loss_ASP = torch.tensor(0.0, device=dev)

        if p_dis_add_noise is not None:
            p_val = float(p_dis_add_noise)
            fake_imcs_g = [_add_noise_prob(x, p=p_val) for x in fake_imcs_g]

        with _autocast_ctx(device, precision):
            fake_score_maps = d((he, fake_imcs_g))
            fake_score_means = [m.mean(dim=(1, 2, 3)) for m in fake_score_maps]
            fake_score_mean = sum(fake_score_means) / len(fake_score_means)
            real_label = torch.ones_like(fake_score_mean, device=dev)
            loss_G = 0.5 * (fake_score_mean - real_label).square().mean()

        loss_G_total = loss_G + loss_L1 + loss_ASP
        if use_scaler and scaler is not None:
            scaler.scale(loss_G_total).backward()
            scaler.step(opt_G)
            if opt_F is not None:
                scaler.step(opt_F)
            scaler.update()
        else:
            loss_G_total.backward()
            opt_G.step()
            if opt_F is not None:
                opt_F.step()

        with torch.no_grad():
            decay = 0.9999 if int(step) >= int(ema_warmup) else 0.0
            for p_ema, p in zip(g_ema.parameters(), g.parameters()):
                p_ema.copy_(p.lerp(p_ema, decay))
            for (b_ema_name, b_ema), (_, b) in zip(g_ema.named_buffers(), g.named_buffers()):
                if "num_batches_tracked" in b_ema_name:
                    b_ema.copy_(b)
                else:
                    b_ema.copy_(b.lerp(b_ema, decay))

        return {
            "loss_D": float(loss_D.detach().item()),
            "loss_R1": float(loss_R1.detach().item()) if isinstance(loss_R1, torch.Tensor) else float(loss_R1),
            "loss_G": float(loss_G.detach().item()),
            "loss_L1": float(loss_L1.detach().item()),
            "loss_ASP": float(loss_ASP.detach().item()) if isinstance(loss_ASP, torch.Tensor) else float(loss_ASP),
            "out_shape": str(tuple(int(s) for s in fake_imc_final.shape)),
            "out_dtype": str(fake_imc_final.dtype),
        }

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(max(int(warmup_steps), 0)):
        _ = train_one_step(step=0)

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    baseline_alloc = int(torch.cuda.memory_allocated(dev)) if device.startswith("cuda") else 0
    baseline_reserved = int(torch.cuda.memory_reserved(dev)) if device.startswith("cuda") else 0

    result = train_one_step(step=0)

    if device.startswith("cuda"):
        torch.cuda.synchronize(dev)
        peak_alloc = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
    else:
        peak_alloc = 0
        peak_reserved = 0

    del g, d, g_ema, opt_G, opt_D, opt_F, he, imc, feats_uni, f_model, e_model, nce_loss, gp_loss
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
    p = argparse.ArgumentParser(description="Measure HistoPlexer training peak CUDA memory (input=3, sweep C).")
    p.add_argument("--device", default="cuda:0", help="Device: cpu / cuda / cuda:N / N (default: cuda:0)")
    p.add_argument("--channels", type=_split_ints, default="3,7,17,28,60", help="Comma-separated C (default: 3,7,17,28,60)")
    p.add_argument("--warmup-steps", type=int, default=1, help="Warmup train steps before measuring (default: 1)")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Autocast precision (default: fp32)")
    p.add_argument("--patch-size", type=int, default=256, help="Output IMC patch size (default: 256)")
    p.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Dummy HE input size. Default: he_patch_size from config, else patch_size*4 if use_high_res else patch_size.",
    )
    p.add_argument("--checkpoint-path", type=Path, default=None, help="Optional checkpoint .pt to read sibling config.json")
    p.add_argument("--base-config", type=Path, default=None, help="Optional config.json to read defaults")

    p.add_argument("--use-high-res", action="store_true", help="Force use_high_res=True (overrides config)")
    p.add_argument("--use-multiscale", action="store_true", help="Force use_multiscale=True (overrides config)")
    p.add_argument("--ngf", type=int, default=None, help="Override ngf")
    p.add_argument("--depth", type=int, default=None, help="Override depth")
    p.add_argument("--encoder-padding", type=int, default=None, help="Override encoder_padding")
    p.add_argument("--decoder-padding", type=int, default=None, help="Override decoder_padding")
    p.add_argument("--fm-feature-size", type=int, default=None, help="Override fm_feature_size")

    p.add_argument("--lr-g", type=float, default=None, help="Override lr_G")
    p.add_argument("--lr-d", type=float, default=None, help="Override lr_D")
    p.add_argument("--lr-f", type=float, default=None, help="Override lr_F")
    p.add_argument("--beta-0", type=float, default=None, help="Override beta_0")
    p.add_argument("--beta-1", type=float, default=None, help="Override beta_1")

    p.add_argument("--use-gp", action="store_true", help="Force use_gp=True (overrides config)")
    p.add_argument("--no-gp", action="store_true", help="Force use_gp=False (overrides config)")
    p.add_argument("--w-l1", type=float, default=None, help="Override w_L1")
    p.add_argument("--w-gp", type=float, default=None, help="Override w_GP")
    p.add_argument("--w-asp", type=float, default=None, help="Override w_ASP")
    p.add_argument("--w-r1", type=float, default=None, help="Override w_R1")
    p.add_argument("--r1-gamma", type=float, default=None, help="Override r1_gamma")
    p.add_argument("--r1-interval", type=int, default=None, help="Override r1_interval")
    p.add_argument("--ema-warmup", type=int, default=None, help="Override ema_warmup")
    p.add_argument("--p-dis-add-noise", type=float, default=None, help="Override p_dis_add_noise (default: config or None)")
    p.add_argument("--blur-gt", action="store_true", help="Force blur_gt=True (overrides config)")

    p.add_argument("--use-feat-enc", action="store_true", help="Force use_feat_enc=True (overrides config)")
    p.add_argument("--no-feat-enc", action="store_true", help="Force use_feat_enc=False (overrides config)")
    p.add_argument("--vgg-path", type=Path, default=None, help="Optional local VGG19 state_dict (no download)")

    p.add_argument(
        "--out-csv",
        type=Path,
        default=_ROOT / "histoplexer_train_peak_memory.csv",
        help="Output CSV path",
    )

    args = p.parse_args(argv)

    cfg = _read_base_config(base_config=args.base_config, checkpoint_path=args.checkpoint_path)

    device, cuda_idx = _parse_device(str(args.device))
    patch_size = int(args.patch_size)

    use_high_res = bool(args.use_high_res) or bool(cfg.get("use_high_res", False))
    use_multiscale = bool(args.use_multiscale) or bool(cfg.get("use_multiscale", True))
    ngf = int(args.ngf if args.ngf is not None else cfg.get("ngf", 32))
    depth = int(args.depth if args.depth is not None else cfg.get("depth", 6))
    encoder_padding = int(args.encoder_padding if args.encoder_padding is not None else cfg.get("encoder_padding", 1))
    decoder_padding = int(args.decoder_padding if args.decoder_padding is not None else cfg.get("decoder_padding", 1))
    fm_feature_size = int(args.fm_feature_size if args.fm_feature_size is not None else cfg.get("fm_feature_size", 0))

    input_size = int(args.input_size) if args.input_size is not None else patch_size
    if args.input_size is None:
        if use_high_res:
            input_size = int(cfg.get("he_patch_size", patch_size * 4))
        else:
            input_size = int(cfg.get("he_patch_size", patch_size))

    lr_G = float(args.lr_g if args.lr_g is not None else cfg.get("lr_G", 0.004))
    lr_D = float(args.lr_d if args.lr_d is not None else cfg.get("lr_D", 2.5e-4))
    lr_F = float(args.lr_f if args.lr_f is not None else cfg.get("lr_F", 0.004))
    beta_0 = float(args.beta_0 if args.beta_0 is not None else cfg.get("beta_0", 0.5))
    beta_1 = float(args.beta_1 if args.beta_1 is not None else cfg.get("beta_1", 0.999))

    if args.use_gp and args.no_gp:
        raise ValueError("Cannot set both --use-gp and --no-gp")
    if args.use_gp:
        use_gp = True
    elif args.no_gp:
        use_gp = False
    else:
        use_gp = bool(cfg.get("use_gp", True))

    w_L1 = float(args.w_l1 if args.w_l1 is not None else cfg.get("w_L1", 1.0))
    w_GP = float(args.w_gp if args.w_gp is not None else cfg.get("w_GP", 0.0))
    w_ASP = float(args.w_asp if args.w_asp is not None else cfg.get("w_ASP", 1.0))
    w_R1 = float(args.w_r1 if args.w_r1 is not None else cfg.get("w_R1", 1.0))
    r1_gamma = float(args.r1_gamma if args.r1_gamma is not None else cfg.get("r1_gamma", 2e-4))
    r1_interval = int(args.r1_interval if args.r1_interval is not None else cfg.get("r1_interval", 16))
    ema_warmup = int(args.ema_warmup if args.ema_warmup is not None else cfg.get("ema_warmup", 5000))
    p_dis_add_noise = args.p_dis_add_noise if args.p_dis_add_noise is not None else cfg.get("p_dis_add_noise", None)
    blur_gt = bool(args.blur_gt) or bool(cfg.get("blur_gt", False))

    if args.use_feat_enc and args.no_feat_enc:
        raise ValueError("Cannot set both --use-feat-enc and --no-feat-enc")
    if args.use_feat_enc:
        use_feat_enc = True
    elif args.no_feat_enc:
        use_feat_enc = False
    else:
        use_feat_enc = bool(cfg.get("use_feat_enc", True))

    vgg_path = args.vgg_path
    if vgg_path is None:
        vgg_from_cfg = cfg.get("vgg_path")
        if isinstance(vgg_from_cfg, str) and vgg_from_cfg.strip():
            vp = Path(vgg_from_cfg)
            if vp.exists():
                vgg_path = vp

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
            c=c,
            device=device,
            cuda_idx=cuda_idx,
            patch_size=patch_size,
            input_size=input_size,
            precision=str(args.precision),
            warmup_steps=int(args.warmup_steps),
            use_high_res=use_high_res,
            use_multiscale=use_multiscale,
            ngf=ngf,
            depth=depth,
            encoder_padding=encoder_padding,
            decoder_padding=decoder_padding,
            fm_feature_size=fm_feature_size,
            lr_G=lr_G,
            lr_D=lr_D,
            lr_F=lr_F,
            beta_0=beta_0,
            beta_1=beta_1,
            use_gp=use_gp,
            w_L1=w_L1,
            w_GP=w_GP,
            w_ASP=w_ASP,
            w_R1=w_R1,
            r1_gamma=r1_gamma,
            r1_interval=r1_interval,
            ema_warmup=ema_warmup,
            p_dis_add_noise=p_dis_add_noise,
            blur_gt=blur_gt,
            use_feat_enc=use_feat_enc,
            vgg_path=vgg_path,
        )

        row = {
            "method": "HistoPlexer",
            "phase": "train",
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
            "use_gp": use_gp,
            "w_L1": w_L1,
            "w_GP": w_GP,
            "w_ASP": w_ASP,
            "w_R1": w_R1,
            "r1_gamma": r1_gamma,
            "r1_interval": r1_interval,
            "ema_warmup": ema_warmup,
            "use_feat_enc": use_feat_enc,
            "vgg_path": str(vgg_path) if vgg_path is not None else "",
            **result,
        }
        rows.append(row)
        print(
            f"[ok] C={c} peak_alloc={row['peak_allocated_mb']:.2f}MB peak_reserved={row['peak_reserved_mb']:.2f}MB "
            f"lossD={row['loss_D']:.4f} lossG={row['loss_G']:.4f} out={row['out_shape']} {row['out_dtype']}"
        )

    print("\n=== Summary (HistoPlexer Training Peak Memory) ===")
    table_cols = ["C", "precision", "peak_allocated_mb", "peak_reserved_mb", "loss_D", "loss_G"]
    formatted = [
        [
            str(r["output_nc"]),
            str(r["precision"]),
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
            "use_gp",
            "w_L1",
            "w_GP",
            "w_ASP",
            "w_R1",
            "r1_gamma",
            "r1_interval",
            "ema_warmup",
            "use_feat_enc",
            "vgg_path",
            "baseline_allocated_mb",
            "baseline_reserved_mb",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "loss_D",
            "loss_R1",
            "loss_G",
            "loss_L1",
            "loss_ASP",
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
