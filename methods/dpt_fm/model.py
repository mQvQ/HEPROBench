from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common import ConvBlock, OutputHead
from methods.pfm import TinyPathologyFoundationEncoder


class DPTFM(nn.Module):
    """Tiny DPT+foundation-model decoder used for the review demo."""

    def __init__(self, encoder_name: str = "hoptimus0", out_channels: int = 4, width: int = 32):
        super().__init__()
        self.encoder = TinyPathologyFoundationEncoder(encoder_name, width=width)
        high_channels = self.encoder.out_channels
        low_channels = self.encoder.low_channels
        self.proj_high = nn.Conv2d(high_channels, width, kernel_size=1)
        self.proj_low = nn.Conv2d(low_channels, width, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBlock(width * 2, width * 2),
            ConvBlock(width * 2, width),
        )
        self.head = OutputHead(width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)
        high = self.proj_high(features["high"])
        high = F.interpolate(high, size=features["low"].shape[-2:], mode="bilinear", align_corners=False)
        low = self.proj_low(features["low"])
        return self.head(self.fuse(torch.cat([low, high], dim=1)))

