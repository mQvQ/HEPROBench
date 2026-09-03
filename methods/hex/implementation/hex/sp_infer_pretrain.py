#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HEX sliding-window inference for the upstream pretrain checkpoint.

This keeps the existing benchmark inference pipeline intact and only replaces
model loading with an explicit two-stage loader:
  1) initialize MUSK backbone via hex_architecture.CustomModel
  2) load the HEX pretrain head checkpoint from config.checkpoint_path
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[1]  # benchmark/methods/HEX
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sp_infer as _base  # noqa: E402
from hex.hex_architecture import CustomModel  # noqa: E402

_MUSK_SOURCE = "hf_hub:xiangjx/musk"
_MUSK_CACHE_PATH = Path.home() / ".cache" / "model.safetensors"


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if not all(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def _unwrap_state(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
        return state
    raise ValueError("checkpoint must be a state_dict or contain model_state_dict/state_dict/model")


def _format_shape(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def _summarize_keys(label: str, keys: List[str], *, limit: int = 8) -> None:
    print(f"  {label}: {len(keys)}")
    if keys:
        shown = ", ".join(keys[:limit])
        suffix = " ..." if len(keys) > limit else ""
        print(f"    {shown}{suffix}")


def _load_model(cfg: Dict[str, Any], *, num_outputs: int, device: torch.device) -> CustomModel:
    print("[LOAD]")
    print("  mode: musk_plus_head")
    print("  musk_source:", _MUSK_SOURCE)
    print("  musk_cache:", str(_MUSK_CACHE_PATH))
    print("  musk_cache_exists:", bool(_MUSK_CACHE_PATH.exists()))
    if _MUSK_CACHE_PATH.exists():
        print("  musk_cache_size_bytes:", int(_MUSK_CACHE_PATH.stat().st_size))

    model = CustomModel(visual_output_dim=1024, num_outputs=int(num_outputs)).to(device)

    ckpt_path = str(cfg["checkpoint_path"])
    raw_state = torch.load(ckpt_path, map_location="cpu")
    state = _strip_module_prefix(_unwrap_state(raw_state))

    model_state = model.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    unexpected_keys: List[str] = []
    mismatched_keys: List[str] = []

    for key, value in state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if _format_shape(model_state[key]) != _format_shape(value):
            mismatched_keys.append(
                f"{key}: ckpt{_format_shape(value)} != model{_format_shape(model_state[key])}"
            )
            continue
        filtered_state[key] = value

    head_weight = filtered_state.get("regression_head1.0.weight")
    if head_weight is not None and int(head_weight.shape[0]) != int(num_outputs):
        raise ValueError(
            "checkpoint output channels do not match channel_names_file: "
            f"ckpt={int(head_weight.shape[0])} cfg={int(num_outputs)}"
        )

    if not filtered_state:
        raise ValueError(
            "No checkpoint tensors matched the pretrain model. "
            "Check checkpoint_path and channel_names_file."
        )

    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing_visual = [k for k in incompatible.missing_keys if k.startswith("visual.")]
    missing_non_visual = [k for k in incompatible.missing_keys if not k.startswith("visual.")]

    print("  head_checkpoint:", ckpt_path)
    print("  checkpoint_tensors:", len(state))
    print("  matched_tensors:", len(filtered_state))
    _summarize_keys("unexpected_tensors", unexpected_keys)
    _summarize_keys("mismatched_tensors", mismatched_keys)
    _summarize_keys("missing_visual_tensors", missing_visual)
    _summarize_keys("missing_non_visual_tensors", missing_non_visual)
    if head_weight is not None:
        print("  checkpoint_output_channels:", int(head_weight.shape[0]))
    print("  configured_output_channels:", int(num_outputs))

    model.eval()
    return model


def main() -> None:
    _base._load_model = _load_model
    _base.main()


if __name__ == "__main__":
    main()
