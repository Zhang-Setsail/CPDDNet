import torch
import torch.nn as nn

class Fusion_Block(nn.Module):
    """
    融合块：使用门控注意力机制融合两个输入
    
    流程：
    1. 两个输入按channel维度拼接
    2. Point-wise Convolution (1x1)
    3. Depth-wise Convolution (3x3)
    4. 按channel拆分为content和gated attention
    5. gated attention经过GELU后与content做Element-wise Multiplication
    6. Point-wise Convolution (1x1)
    7. 与原始第一个输入做Element-wise Addition
    """
    def __init__(self, in_channels=3):
        super(Fusion_Block, self).__init__()
        self.in_channels = in_channels
        
        # Point-wise Convolution (1x1) - 输入是拼接后的2*in_channels
        self.pointwise_conv1 = nn.Conv2d(
            in_channels * 2, in_channels * 2, 
            kernel_size=1, stride=1, padding=0
        )
        
        # Depth-wise Convolution (3x3) - 每个通道独立卷积
        self.depthwise_conv = nn.Conv2d(
            in_channels * 2, in_channels * 2, 
            kernel_size=3, stride=1, padding=1, 
            groups=in_channels * 2  # 关键：groups参数实现深度卷积
        )
        
        # Point-wise Convolution (1x1) - 最终输出
        self.pointwise_conv2 = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=1, stride=1, padding=0
        )
        
        # 激活函数
        self.gelu = nn.GELU()
    
    def forward(self, x1, x2):
        """
        前向传播
        Args:
            x1: 第一个输入（去噪之后）- shape: (B, C, H, W)
            x2: 第二个输入（原始图像）- shape: (B, C, H, W)
        Returns:
            融合后的输出 - shape: (B, C, H, W)
        """
        # 保存原始第一个输入用于最后的残差连接
        residual = x1
        
        # 1. 按channel维度拼接两个输入
        concat_input = torch.cat([x1, x2], dim=1)  # shape: (B, 2*C, H, W)
        
        # 2. Point-wise Convolution (1x1)
        x = self.pointwise_conv1(concat_input)  # shape: (B, 2*C, H, W)
        
        # 3. Depth-wise Convolution (3x3)
        x = self.depthwise_conv(x)  # shape: (B, 2*C, H, W)
        
        # 4. 按channel拆分为content和gated attention
        content, gated_attention = torch.split(x, self.in_channels, dim=1)
        # content: (B, C, H, W)
        # gated_attention: (B, C, H, W)
        
        # 5. gated attention经过GELU后与content做Element-wise Multiplication
        gated_attention = self.gelu(gated_attention)
        fused_features = content * gated_attention  # shape: (B, C, H, W)
        
        # 6. Point-wise Convolution (1x1)
        output = self.pointwise_conv2(fused_features)  # shape: (B, C, H, W)
        
        # 7. 与原始第一个输入做Element-wise Addition（残差连接）
        final_output = output + residual  # shape: (B, C, H, W)
        
        return final_output

class Fusion_UNet(nn.Module):
    """
    双UNet融合网络，使用共享的编码器和解码器处理两个不同输入
    
    Args:
        nf: 基础特征通道数，默认64
        in_channels: 输入通道数，默认3
        out_channels: 输出通道数，默认3
    
    特点:
        - 共享编码器：两个输入使用相同的编码器网络参数
        - 共享解码器：两个编码特征使用相同的解码器网络参数
        - 融合输出：两个分支的输出相加后作为最终结果
    """
    
    def __init__(self, nf=64, in_channels=3, out_channels=3):
        super(Fusion_UNet, self).__init__()
        
        # 共享编码器 (Shared Encoder)
        # 输入卷积层
        self.inc_conv1 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.inc_conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        
        # 下采样层
        self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1_conv1 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1)
        self.down1_conv2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        
        self.down2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2_conv1 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.down2_conv2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        
        self.down3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3_conv1 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1)
        self.down3_conv2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        
        self.down4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4_conv1 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        self.down4_conv2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        
        # 共享解码器 (Shared Decoder)
        # 第1层上采样: 512+512=1024 -> 256  
        self.shared_up1_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.shared_up1_conv1 = nn.Conv2d(nf * 8 * 2, nf * 8, kernel_size=3, stride=1, padding=1)
        self.shared_up1_conv2 = nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1)
        
        # 第2层上采样: 256+256 -> 128
        self.shared_up2_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.shared_up2_conv1 = nn.Conv2d(nf * 4 * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.shared_up2_conv2 = nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1)
        
        # 第3层上采样: 128+128 -> 64
        self.shared_up3_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.shared_up3_conv1 = nn.Conv2d(nf * 2 * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        self.shared_up3_conv2 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        
        # 第4层上采样: 64+64 -> 64
        self.shared_up4_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.shared_up4_conv1 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        self.shared_up4_conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        # 输出卷积层 - 为每个UNet分支生成独立输出
        self.outc = nn.Conv2d(nf, out_channels, kernel_size=3, stride=1, padding=1)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        self.fusion_block_1 = Fusion_Block(in_channels=nf)
        self.fusion_block_2 = Fusion_Block(in_channels=nf*2)
        self.fusion_block_3 = Fusion_Block(in_channels=nf*4)
        self.fusion_block_4 = Fusion_Block(in_channels=nf*8)
        self.fusion_block_5 = Fusion_Block(in_channels=nf*8)
        
    def encoder_forward(self, x):
        """共享编码器的前向传播"""
        # 输入层
        x1 = self.relu(self.inc_conv1(x))
        x1 = self.relu(self.inc_conv2(x1))
        
        # 下采样层
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
        
        return x1, x2, x3, x4, x5
    
    def shared_decoder_forward(self, x5, x4, x3, x2, x1):
        """共享解码器的前向传播"""
        # 上采样层
        x = self.shared_up1_upconv(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.relu(self.shared_up1_conv1(x))
        x = self.relu(self.shared_up1_conv2(x))
        
        x = self.shared_up2_upconv(x)
        x = torch.cat([x3, x], dim=1)
        x = self.relu(self.shared_up2_conv1(x))
        x = self.relu(self.shared_up2_conv2(x))
        
        x = self.shared_up3_upconv(x)
        x = torch.cat([x2, x], dim=1)
        x = self.relu(self.shared_up3_conv1(x))
        x = self.relu(self.shared_up3_conv2(x))
        
        x = self.shared_up4_upconv(x)
        x = torch.cat([x1, x], dim=1)
        x = self.relu(self.shared_up4_conv1(x))
        x = self.relu(self.shared_up4_conv2(x))
        
        # 使用指定的输出层
        out = self.outc(x)
        return out
    
    def forward(self, x1, x2):
        """
        前向传播
        Args:
            x1: 第一个输入——去噪之后
            x2: 第二个输入——原始图像
        """
        # 保存输入用于全局残差连接（使用第一个输入）
        input_x = x1
        
        # 使用共享编码器处理第一个输入
        enc1_x1, enc1_x2, enc1_x3, enc1_x4, enc1_x5 = self.encoder_forward(x1)
        
        # 使用共享编码器处理第二个输入
        enc2_x1, enc2_x2, enc2_x3, enc2_x4, enc2_x5 = self.encoder_forward(x2)
        
        # 将两个编码器的输出在每一层相加
        fused_x1 = self.fusion_block_1(enc1_x1, enc2_x1)
        fused_x2 = self.fusion_block_2(enc1_x2, enc2_x2)
        fused_x3 = self.fusion_block_3(enc1_x3, enc2_x3)
        fused_x4 = self.fusion_block_4(enc1_x4, enc2_x4)
        fused_x5 = self.fusion_block_5(enc1_x5, enc2_x5)
        
        # 使用共享解码器处理融合后的特征（使用outc1作为输出层）
        out = self.shared_decoder_forward(fused_x5, fused_x4, fused_x3, fused_x2, fused_x1)
        
        # 全局残差连接
        final_out = out + input_x
        
        return final_out