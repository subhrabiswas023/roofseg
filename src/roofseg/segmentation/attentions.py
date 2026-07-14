import math

from torch import nn
import torch


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        # 1. The Factorized 1D Pooling (The core innovation)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Collapses Width, keeps Height
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Collapses Height, keeps Width
        
        # Calculate bottleneck size (min of 8 channels to prevent extreme crushing)
        mip = max(8, channels // reduction)
        
        # 2. Shared Bottleneck (MLP)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU() # SiLU (Swish) is standard for modern attention
        
        # 3. Independent Output Projections
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        
        # Squeeze
        x_h = self.pool_h(x)
        # Permute x_w so we can concatenate it along the spatial height dimension
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Excitation (Shared)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back into X and Y strips
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) # Permute back to original orientation
        
        # Scale (Generate Masks)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # Apply the intersecting masks to the original feature map
        return identity * a_w * a_h

class LargeKernelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # 1. Local Spatial Context (5x5 Depthwise)
        self.conv_local = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
        
        # 2. Long-Range Spatial Context (5x5 Depthwise with Dilation 3)
        # padding = (kernel_size - 1) / 2 * dilation = (5 - 1) / 2 * 3 = 6
        self.conv_dilated = nn.Conv2d(
            channels, channels, kernel_size=5, stride=1, padding=6, 
            groups=channels, dilation=3
        )
        
        # 3. Channel Mixing (1x1 Pointwise)
        self.conv_channel = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        
        # The sequential decomposition pipeline
        attn = self.conv_local(x)
        attn = self.conv_dilated(attn)
        attn = self.conv_channel(attn)
        
        return u * attn

class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, b: int = 1, gamma: int = 2):
        super().__init__()
        # Dynamically calculate adaptive kernel size based on channel depth
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 != 0 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling to isolate channel descriptors
        y = self.avg_pool(x)
        
        # Reshape for 1D convolution processing: [Batch, 1, Channels]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Scale original features by channel importance weights
        return x * y.sigmoid()