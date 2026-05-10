import torch.nn as nn
from . import register_model
from .base_model import BaseModel
import torch

@register_model('SubModelDnRgb2Rgb')
class SubModelDnRgb2Rgb(BaseModel):
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

        dn_in_nc = config.get('in_nc', 16)
        dn_out_nc = config.get('out_nc', 16)
        dn_nf = config.get('nf', 32)

        self.dn_conv1_1 = nn.Conv2d(dn_in_nc, dn_nf, kernel_size=3, stride=1, padding=1)
        self.dn_conv1_2 = nn.Conv2d(dn_nf, dn_nf, kernel_size=3, stride=1, padding=1)
        self.dn_pool1 = nn.MaxPool2d(kernel_size=2)

        self.dn_conv2_1 = nn.Conv2d(dn_nf, dn_nf * 2, kernel_size=3, stride=1, padding=1)
        self.dn_conv2_2 = nn.Conv2d(dn_nf * 2, dn_nf * 2, kernel_size=3, stride=1, padding=1)
        self.dn_pool2 = nn.MaxPool2d(kernel_size=2)

        self.dn_conv3_1 = nn.Conv2d(dn_nf * 2, dn_nf * 4, kernel_size=3, stride=1, padding=1)
        self.dn_conv3_2 = nn.Conv2d(dn_nf * 4, dn_nf * 4, kernel_size=3, stride=1, padding=1)
        self.dn_pool3 = nn.MaxPool2d(kernel_size=2)

        self.dn_conv4_1 = nn.Conv2d(dn_nf * 4, dn_nf * 8, kernel_size=3, stride=1, padding=1)
        self.dn_conv4_2 = nn.Conv2d(dn_nf * 8, dn_nf * 8, kernel_size=3, stride=1, padding=1)

        self.dn_upv5 = nn.ConvTranspose2d(dn_nf * 8, dn_nf * 4, 2, stride=2)
        self.dn_conv5_1 = nn.Conv2d(dn_nf * 8, dn_nf * 4, kernel_size=3, stride=1, padding=1)
        self.dn_conv5_2 = nn.Conv2d(dn_nf * 4, dn_nf * 4, kernel_size=3, stride=1, padding=1)

        self.dn_upv6 = nn.ConvTranspose2d(dn_nf * 4, dn_nf * 2, 2, stride=2)
        self.dn_conv6_1 = nn.Conv2d(dn_nf * 4, dn_nf * 2, kernel_size=3, stride=1, padding=1)
        self.dn_conv6_2 = nn.Conv2d(dn_nf * 2, dn_nf * 2, kernel_size=3, stride=1, padding=1)

        self.dn_upv7 = nn.ConvTranspose2d(dn_nf * 2, dn_nf, 2, stride=2)
        self.dn_conv7_1 = nn.Conv2d(dn_nf * 2, dn_nf, kernel_size=3, stride=1, padding=1)
        self.dn_conv7_2 = nn.Conv2d(dn_nf, dn_nf, kernel_size=3, stride=1, padding=1)

        self.dn_conv8_1 = nn.Conv2d(dn_nf, dn_out_nc, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 保存输入用于全局残差连接
        input_x = x
        
        # 编码器第1层
        dn_conv1 = self.relu(self.dn_conv1_1(x))
        # dn_conv1_identity = dn_conv1
        dn_conv1 = self.relu(self.dn_conv1_2(dn_conv1))
        # dn_conv1 = dn_conv1 + dn_conv1_identity  # 残差连接
        dn_pool1 = self.dn_pool1(dn_conv1)

        # 编码器第2层
        dn_conv2 = self.relu(self.dn_conv2_1(dn_pool1))
        # dn_conv2_identity = dn_conv2
        dn_conv2 = self.relu(self.dn_conv2_2(dn_conv2))
        # dn_conv2 = dn_conv2 + dn_conv2_identity  # 残差连接
        dn_pool2 = self.dn_pool2(dn_conv2)

        # 编码器第3层
        dn_conv3 = self.relu(self.dn_conv3_1(dn_pool2))
        # dn_conv3_identity = dn_conv3
        dn_conv3 = self.relu(self.dn_conv3_2(dn_conv3))
        # dn_conv3 = dn_conv3 + dn_conv3_identity  # 残差连接
        dn_pool3 = self.dn_pool3(dn_conv3)

        # 编码器第4层（现在是底部）
        dn_conv4 = self.relu(self.dn_conv4_1(dn_pool3))
        # dn_conv4_identity = dn_conv4
        dn_conv4 = self.relu(self.dn_conv4_2(dn_conv4))
        # dn_conv4 = dn_conv4 + dn_conv4_identity  # 残差连接

        # 解码器第5层
        dn_up5 = self.dn_upv5(dn_conv4)
        dn_up5 = torch.cat([dn_up5, dn_conv3], 1)
        dn_conv5 = self.relu(self.dn_conv5_1(dn_up5))
        # dn_conv5_identity = dn_conv5
        dn_conv5 = self.relu(self.dn_conv5_2(dn_conv5))
        # dn_conv5 = dn_conv5 + dn_conv5_identity  # 残差连接

        # 解码器第6层
        dn_up6 = self.dn_upv6(dn_conv5)
        dn_up6 = torch.cat([dn_up6, dn_conv2], 1)
        dn_conv6 = self.relu(self.dn_conv6_1(dn_up6))
        # dn_conv6_identity = dn_conv6
        dn_conv6 = self.relu(self.dn_conv6_2(dn_conv6))
        # dn_conv6 = dn_conv6 + dn_conv6_identity  # 残差连接

        # 解码器第7层
        dn_up7 = self.dn_upv7(dn_conv6)
        dn_up7 = torch.cat([dn_up7, dn_conv1], 1)
        dn_conv7 = self.relu(self.dn_conv7_1(dn_up7))
        # dn_conv7_identity = dn_conv7
        dn_conv7 = self.relu(self.dn_conv7_2(dn_conv7))
        # dn_conv7 = dn_conv7 + dn_conv7_identity  # 残差连接

        # 输出层
        out = self.dn_conv8_1(dn_conv7)
        
        # 全局残差连接（输入和输出通道数相同时）
        out = out + input_x  # 全局残差连接

        return out

    def get_model_info(self):
        pass 