from __future__ import annotations

from typing import Any

import torch

from .dpt_fm.model import DPTFM
from .cut.model import CUTAdapter
from .gigatime.model import GigaTIMEAdapter
from .hex.model import HEXAdapter
from .pytorch_cyclegan_and_pix2pix.model import PyTorchCycleGANAndPix2PixAdapter
from .rosie.model import ROSIEAdapter
from .miphei_vit.model import MIPHEIViT


def build_method(method_cfg: dict[str, Any], out_channels: int) -> torch.nn.Module:
    name = str(method_cfg.get("name", "")).lower()
    encoder_name = str(method_cfg.get("encoder_name", "hoptimus0")).lower()
    width = int(method_cfg.get("width", 32))
    if name in {"miphei_vit", "miphei-vit", "vitmatte"}:
        return MIPHEIViT(encoder_name=encoder_name, out_channels=out_channels, width=width)
    if name in {"dpt_fm", "dpt"}:
        return DPTFM(encoder_name=encoder_name, out_channels=out_channels, width=width)
    if name == "cut":
        return CUTAdapter(out_channels=out_channels, width=width)
    if name == "hex":
        return HEXAdapter(out_channels=out_channels, width=width)
    if name == "gigatime":
        return GigaTIMEAdapter(out_channels=out_channels, width=width)
    if name in {"pix2pix", "cycle_gan", "pytorch_cyclegan_and_pix2pix", "pytorch-cyclegan-and-pix2pix"}:
        variant = str(method_cfg.get("variant", name))
        return PyTorchCycleGANAndPix2PixAdapter(out_channels=out_channels, width=width, variant=variant)
    if name == "rosie":
        return ROSIEAdapter(out_channels=out_channels, width=width)
    raise ValueError(f"Unknown method name: {name}")

