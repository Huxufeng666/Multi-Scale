
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Decouple import DecoupleLayer
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 2. 您的 ChannelAttention 模块 (保持不变)
# -----------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# -----------------------------------------------------------------
# (!! 已修改 !!) 3. 重写的 SpatialAttention 模块
# (这个新版本会融合 B 和 F，并输出 1x 通道数)
# -----------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # 1. 1x1 卷积，用于将 B 和 F 融合 (2*C -> C)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        
        # 2. 标准空间注意力 (在融合后的特征上操作)
        self.conv_sa = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, B, F):
        # 1. 融合 B 和 F
        O_cat = torch.cat([B, F], dim=1) # (B, 2*C, H, W)
        O_fused = self.fuse_conv(O_cat)  # (B, C, H, W)
        
        # 2. 在融合后的特征上计算空间注意力
        avg_out = torch.mean(O_fused, dim=1, keepdim=True)
        max_out, _ = torch.max(O_fused, dim=1, keepdim=True)
        M_in = torch.cat([avg_out, max_out], dim=1)
        
        M = self.sigmoid(self.conv_sa(M_in)) # (B, 1, H, W)
        
        # 3. 将注意力应用到融合后的特征上
        O_prime = O_fused * M
        return O_prime

# -----------------------------------------------------------------
# (!! 新模块 !!) 4. 您的新流程模块

class EfficientFeature(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        total_in = sum(in_channels_list)
        self.consolidator = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.channel_att = ChannelAttention(out_channels)
        self.decouple = DecoupleLayer(out_channels)
        self.spatial_att = SpatialAttention(out_channels)

    def forward(self, inputs, mask=None):
        # inputs: list of tensors [x1, x2, x3, x4]
        x_cat = torch.cat(inputs, dim=1)
        x_cons = self.consolidator(x_cat)
        x_ca = self.channel_att(x_cons)
        B, F, loss_b, loss_f = self.decouple(x_ca, mask)
        out = self.spatial_att(B, F)
        return out, loss_b, loss_f















# -----------------------------------------------------------------
# 5. 修正后的测试代码
# -----------------------------------------------------------------
if __name__ == "__main__":
    pass
    # # 1. 创建 4 个 (B, 1024, 16, 16) 的输入
    # x1 = torch.randn(2, 1024, 16, 16) 
    # x2 = torch.randn(2, 1024, 16, 16)
    # x3 = torch.randn(2, 1024, 16, 16)
    # x4 = torch.randn(2, 1024, 16, 16)
    # mask = torch.randn(2, 1, 16, 16)
    
    # print(f"--- Testing FeatureFusionPipeline ---")
    
    # # 2. 初始化新模块
    # model = EfficientFeature(in_channels=1024)
    
    # # 3. 运行
    # out, loss_b, loss_f = model(x1, x2, x3, x4, mask)
    
    # # 4. 检查输出
    # print(f"Input shape (x4): {x4.shape}")
    # print(f"Final Output shape: {out.shape}")
    # print(f"Loss B: {loss_b.item():.4f}")
    # print(f"Loss F: {loss_f.item():.4f}")

    # # 5. 验证
    # assert out.shape == (2, 1024, 16, 16)
    # print("✅ Test successful! Output shape is correct (B, C, H, W).")