from __future__ import annotations

from methods.simple_adapters import SimpleAdapterMethod


class HEXAdapter(SimpleAdapterMethod):
    def __init__(self, out_channels: int, width: int = 32):
        super().__init__(method_name="hex", out_channels=out_channels, width=width)

