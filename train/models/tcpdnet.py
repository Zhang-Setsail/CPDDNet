import torch
from torch import nn

from . import bilinear
from .base_unet import UNet_Base


class TCPDNet(nn.Module):
    """Stage I color and polarization demosaicking backbone."""

    def __init__(self, nf=64):
        super().__init__()
        self.color_unet = UNet_Base(nf=nf, in_channels=3, out_channels=3)
        self.polar_unet = UNet_Base(nf=nf, in_channels=4, out_channels=4)

    def forward(self, raw):
        sub_bayer = bilinear.generate_sub_bayer(raw)
        colors = torch.stack(
            [self.color_unet(bilinear.Color_Bilinear(sub_bayer[:, i])) for i in range(4)],
            dim=1,
        )
        polar_mosaic = bilinear.generate_polar_mosaic(colors)
        polars = torch.stack(
            [self.polar_unet(bilinear.Polar_Bilinear(polar_mosaic[:, i])) for i in range(3)],
            dim=2,
        )
        return polars, colors
