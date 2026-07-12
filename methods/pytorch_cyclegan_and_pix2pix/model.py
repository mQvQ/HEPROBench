from __future__ import annotations

from methods.simple_adapters import SimpleAdapterMethod


class PyTorchCycleGANAndPix2PixAdapter(SimpleAdapterMethod):
    def __init__(self, out_channels: int, width: int = 32, variant: str = "pix2pix"):
        method_name = "cycle_gan" if variant == "cycle_gan" else "pix2pix"
        super().__init__(method_name=method_name, out_channels=out_channels, width=width)
        self.variant = variant

