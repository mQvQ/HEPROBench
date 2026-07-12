from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from methods.common import ConvBlock
from .registry import get_pathology_foundation_model


class TinyPathologyFoundationEncoder(nn.Module):
    """Small stand-in for external pathology foundation model encoders."""

    def __init__(self, encoder_name: str, width: int = 32):
        super().__init__()
        self.encoder_name = encoder_name.lower()
        self.display_name = get_pathology_foundation_model(self.encoder_name)
        offset = int(hashlib.sha1(self.encoder_name.encode("utf-8")).hexdigest()[:2], 16) % 8
        stem_width = int((width + offset + 3) // 4 * 4)
        self.low_channels = stem_width
        self.out_channels = stem_width * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, kernel_size=3, padding=1),
            nn.GroupNorm(4, stem_width),
            nn.GELU(),
        )
        self.block1 = ConvBlock(stem_width, stem_width)
        self.down = nn.Conv2d(stem_width, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.block2 = ConvBlock(self.out_channels, self.out_channels)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        low = self.block1(self.stem(x))
        high = self.block2(self.down(low))
        return {"low": low, "high": high}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["high"]

