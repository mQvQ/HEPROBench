"""Model construction helpers for GRPO FM training."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Dict, Tuple

import torch

from .generators import get_generator


def _make_generator_cfg(model_cfg: Dict) -> SimpleNamespace:
    encoder = SimpleNamespace(
        encoder_name=str(model_cfg.get("encoder_name", "hoptimus0")),
        encoder_weights=model_cfg.get("encoder_weights", None),
        frozen=bool(model_cfg.get("frozen_encoder", False)),
    )
    model = SimpleNamespace(
        model_name=str(model_cfg.get("model_name", "myvitmatte")),
        encoder=encoder,
        use_lora=bool(model_cfg.get("use_lora", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    train = SimpleNamespace(foreground_head=False)
    return SimpleNamespace(model=model, train=train)


def _extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        for key in ("state_dict", "model", "generator", "generator_state_dict"):
            if key in raw_obj and isinstance(raw_obj[key], dict):
                return raw_obj[key]
    if isinstance(raw_obj, dict):
        return raw_obj
    raise TypeError(f"Unsupported checkpoint object type: {type(raw_obj)}")


def _sanitize_generator_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            k = k[len("generator.") :]
        elif k.startswith("model.generator."):
            k = k[len("model.generator.") :]
        clean[k] = v
    return clean


def load_generator_checkpoint(generator: torch.nn.Module, checkpoint_path: str) -> Tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _sanitize_generator_keys(state_dict)
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    return len(missing), len(unexpected)


def build_grpo_generator(
    model_cfg: Dict,
    img_size: int,
    in_channels: int,
    out_channels: int,
) -> torch.nn.Module:
    cfg = _make_generator_cfg(model_cfg)
    generator = get_generator(
        cfg.model.model_name,
        img_size=img_size,
        nc_in=in_channels,
        nc_out=out_channels,
        cfg=cfg,
    )

    ckpt_path = model_cfg.get("checkpoint_path", None)
    if ckpt_path:
        missing, unexpected = load_generator_checkpoint(generator, str(ckpt_path))
        print(
            f"[info] loaded checkpoint: {ckpt_path} "
            f"(missing_keys={missing}, unexpected_keys={unexpected})"
        )

    return generator


def build_reference_model(train_model: torch.nn.Module) -> torch.nn.Module:
    reference = copy.deepcopy(train_model)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False
    return reference
