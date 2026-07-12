from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from .common import ConvBlock, OutputHead


class SimpleAdapterMethod(nn.Module):
    """Tiny runnable adapters for external benchmark methods."""

    def __init__(self, method_name: str, out_channels: int, width: int = 32):
        super().__init__()
        self.method_name = method_name
        offset = int(hashlib.sha1(method_name.encode("utf-8")).hexdigest()[:2], 16) % 8
        w = int((width + offset + 3) // 4 * 4)
        if method_name == "hex":
            depth = 1
        elif method_name in {"cut", "pix2pix", "cycle_gan"}:
            depth = 3
        elif method_name == "gigatime":
            depth = 4
        elif method_name == "rosie":
            depth = 2
        else:
            depth = 2
        layers: list[nn.Module] = [
            nn.Conv2d(3, w, kernel_size=3, padding=1),
            nn.GroupNorm(4, w),
            nn.GELU(),
        ]
        for _ in range(depth):
            layers.append(ConvBlock(w, w))
        self.trunk = nn.Sequential(*layers)
        self.head = OutputHead(w, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

