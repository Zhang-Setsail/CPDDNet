import torch
from torch import nn

from . import bilinear
from .base_unet import UNet_Base
from .fusion_unet import Fusion_UNet
from .sub_model_Dn import SubModelDn


def unpack_raw(packed):
    batch, channels, height, width = packed.shape
    if channels != 16:
        raise ValueError(f"Expected 16 packed RAW channels, got {channels}")
    return packed.reshape(batch, 4, 4, height, width).permute(0, 3, 1, 4, 2).reshape(batch, 1, height * 4, width * 4)


class PIDNDM(nn.Module):
    """Stage II network with frozen DN/DM backbones and trainable GFMs."""

    def __init__(self, nf=64):
        super().__init__()
        self.dn_model = SubModelDn({"in_nc": 16, "out_nc": 16, "nf": nf})
        self.color_unet = Fusion_UNet(nf=nf, in_channels=3, out_channels=3)
        self.polar_unet = UNet_Base(nf=nf, in_channels=4, out_channels=4)

    def forward(self, packed_noisy):
        packed_denoised = self.dn_model(packed_noisy)
        denoised = bilinear.generate_sub_bayer(unpack_raw(packed_denoised))
        noisy = bilinear.generate_sub_bayer(unpack_raw(packed_noisy))
        colors = torch.stack(
            [
                self.color_unet(
                    bilinear.Color_Bilinear(denoised[:, i]),
                    bilinear.Color_Bilinear(noisy[:, i]),
                )
                for i in range(4)
            ],
            dim=1,
        )
        polar_mosaic = bilinear.generate_polar_mosaic(colors)
        polars = torch.stack(
            [self.polar_unet(bilinear.Polar_Bilinear(polar_mosaic[:, i])) for i in range(3)],
            dim=2,
        )
        return polars, colors
