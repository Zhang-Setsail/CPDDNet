import torch.nn as nn
from . import register_model
from .base_model import BaseModel
import torch

import torch.nn as nn
from . import register_model
from .base_model import BaseModel
import torch
from .base_unet import UNet_Base
from .fusion_unet import Fusion_UNet
from . import bilinear
from utils import shape_control

from .sub_model_Dn import SubModelDn

@register_model('PIDNDM')
class PIDNDM(BaseModel):
    """
    SubModelDn for denoising network with residual connections
    
    Args:
        config: 配置字典，包含网络参数
            - in_nc: 输入通道数，默认16
            - out_nc: 输出通道数，默认16
            - nf: 基础特征数，默认32
    """
    
    def __init__(self, config):
        super().__init__(config)

        self.dn_model = SubModelDn(config['dn_model'])

        self.color_unet = Fusion_UNet(nf=config['dm_model']['nf'], in_channels=3, out_channels=3)
        self.polar_unet = UNet_Base(nf=config['dm_model']['nf'], in_channels=4, out_channels=4)
        
        self.colors = None
        self.polars = None

    def forward(self, x):
        # 保存输入用于全局残差连接
        input_x = x

        dn_out = self.dn_model(input_x)

        dn_out_raw = shape_control.channel_separate_to_dense_raw_torch(dn_out).unsqueeze(1)
        dn_in_raw = shape_control.channel_separate_to_dense_raw_torch(input_x).unsqueeze(1)

        sub_bayer_x = bilinear.generate_sub_bayer(dn_out_raw)
        sub_bayer_x_noise = bilinear.generate_sub_bayer(dn_in_raw)

        self.colors = []
        self.polars = []

        for i in range(4):
            sub_color_x = bilinear.Color_Bilinear(sub_bayer_x[::,i,::,::])
            sub_color_x_noise = bilinear.Color_Bilinear(sub_bayer_x_noise[::,i,::,::])
            sub_color_x = self.color_unet(sub_color_x, sub_color_x_noise)
            self.colors.append(sub_color_x)

        colors = torch.stack(self.colors, dim=1)

        pixel_shuffle = bilinear.generate_polar_mosaic(colors)

        for i in range(3):
            sub_polar = bilinear.Polar_Bilinear(pixel_shuffle[::,i,::,::,::])
            sub_polar = self.polar_unet(sub_polar)
            self.polars.append(sub_polar)

        out = torch.stack(self.polars, dim=2)

        return out, colors

    def get_model_info(self):
        pass 