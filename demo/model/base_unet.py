import torch
import torch.nn as nn

class UNet_Base(nn.Module):
    """
    基于unet_default.yaml配置生成的UNet网络
    
    Args:
        config_path: YAML配置文件路径，默认为'unet_default.yaml'
        in_channels: 输入通道数，默认3
        out_channels: 输出通道数，默认1
    """
    
    def __init__(self, nf=64, in_channels=3, out_channels=3):
        super(UNet_Base, self).__init__()
        
        # 输入卷积层 (inc)
        self.inc_conv1 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.inc_conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        
        # 编码器下采样层 (down)
        # 第1层: 64 -> 128
        self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1_conv1 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1)
        self.down1_conv2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        
        # 第2层: 128 -> 256
        self.down2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2_conv1 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.down2_conv2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        
        # 第3层: 256 -> 512
        self.down3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3_conv1 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1)
        self.down3_conv2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        
        # 第4层: 512 -> 1024
        self.down4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4_conv1 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        self.down4_conv2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        
        # 解码器上采样层 (up)
        # 第1层: 512+512=1024 -> 256
        self.up1_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up1_conv1 = nn.Conv2d(nf * 8 * 2, nf * 8, kernel_size=3, stride=1, padding=1)
        self.up1_conv2 = nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1)
        
        # 第2层: 256+256 -> 128
        self.up2_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2_conv1 = nn.Conv2d(nf * 4 * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.up2_conv2 = nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1)
        
        # 第3层: 128+128 -> 64
        self.up3_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3_conv1 = nn.Conv2d(nf * 2 * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        self.up3_conv2 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        
        # 第4层: 64+64 -> 64
        self.up4_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up4_conv1 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        self.up4_conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        # 输出卷积层 (outc) - 无激活函数
        self.outc = nn.Conv2d(nf, out_channels, kernel_size=3, stride=1, padding=1)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 保存输入用于全局残差连接
        input_x = x
        
        x1 = self.relu(self.inc_conv1(x))
        x1 = self.relu(self.inc_conv2(x1))
        
        x2 = self.down1_pool(x1)
        x2 = self.relu(self.down1_conv1(x2))
        x2 = self.relu(self.down1_conv2(x2))
        
        
        x3 = self.down2_pool(x2)
        x3 = self.relu(self.down2_conv1(x3))
        x3 = self.relu(self.down2_conv2(x3))
        
        x4 = self.down3_pool(x3)
        x4 = self.relu(self.down3_conv1(x4))
        x4 = self.relu(self.down3_conv2(x4))
        
        x5 = self.down4_pool(x4)
        x5 = self.relu(self.down4_conv1(x5))
        x5 = self.relu(self.down4_conv2(x5))

        x = self.up1_upconv(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.relu(self.up1_conv1(x))
        x = self.relu(self.up1_conv2(x))
        
        x = self.up2_upconv(x)
        x = torch.cat([x3, x], dim=1)
        x = self.relu(self.up2_conv1(x))
        x = self.relu(self.up2_conv2(x))
        
        x = self.up3_upconv(x)
        x = torch.cat([x2, x], dim=1)
        x = self.relu(self.up3_conv1(x))
        x = self.relu(self.up3_conv2(x))

        x = self.up4_upconv(x)
        x = torch.cat([x1, x], dim=1)
        x = self.relu(self.up4_conv1(x))
        x = self.relu(self.up4_conv2(x))
        
        out = self.outc(x)

        out = out + input_x  # 全局残差连接
        return out