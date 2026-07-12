from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common import ConvBlock, OutputHead
from methods.pfm import TinyPathologyFoundationEncoder


class MIPHEIViT(nn.Module):
    """Tiny MIPHEI-ViT demo: H-Optimus-style FM encoder + ViTMatte-like decoder."""

    def __init__(self, encoder_name: str = "hoptimus0", out_channels: int = 4, width: int = 32):
        super().__init__()
        self.encoder = TinyPathologyFoundationEncoder(encoder_name, width=width)
        high_channels = self.encoder.out_channels
        self.decoder_high = ConvBlock(high_channels, width * 2)
        self.decoder_low = ConvBlock(width * 2 + self.encoder.low_channels, width)
        self.head = OutputHead(width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)
        y = self.decoder_high(features["high"])
        y = F.interpolate(y, size=features["low"].shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, features["low"]], dim=1)
        y = self.decoder_low(y)
        return self.head(y)

