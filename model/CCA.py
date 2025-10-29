import torch
import torch.nn as nn
import torch.nn.functional as F

class CristCoistAttention(nn.Module):
    """
    Crist-Coist Attention 模块：
    - Crist: 通道注意力 (Channel refinement)
    - Coist: 空间注意力 (Spatial interaction)
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CristCoistAttention, self).__init__()

        # 🔸 Crist 分支 - 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # 🔹 Coist 分支 - 空间注意力
        padding = (kernel_size - 1) // 2
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- Crist 分支 ---
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        crist = x * y  # 通道增强

        # --- Coist 分支 ---
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        coist = torch.cat([avg_out, max_out], dim=1)
        coist = self.spatial(coist)  # 空间注意力 mask
        coist = x * coist

        # --- 融合输出 ---
        out = x + crist + coist
        return out


class CCA_FusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        total_ch = sum(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_ch, channels[-1], kernel_size=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=False )
        )

    def forward(self, features):
        # features: list of feature maps from [cca1, cca2, cca3, block4]
        # 需要 resize 到相同尺度 (block4 尺度)
        target_size = features[-1].shape[2:]
        resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=True)
                   for f in features]
        fused = torch.cat(resized, dim=1)
        return self.fuse(fused)


if __name__ == "__main__":
    pass
    # # 创建随机输入
    # # x = torch.randn(16, 64, 256, 256)  # batch=2, 单通道, 256x256
    # x2 = torch.randn(16, 64, 128, 128)  # batch=2, 单通道, 256x256
    # x3 = torch.randn(16, 128, 64, 64)  # batch=2, 单通道, 256x256
    # x4 = torch.randn(16, 256, 32, 32)  # batch=2, 单通道, 256x256
    # x5 = torch.randn(16, 1024, 16, 16)  # batch=2, 单通道, 256x256

    # channel_list = [64, 128, 256, 1024]
    # model = CCA_FusionModule(channels=channel_list)
    # # 创建模型（是否启用 decouple 模块）
    # features_list = [x2, x3, x4, x5]
    
    # # 前向传播测试
    # with torch.no_grad():
    #     y = model(features_list)
    #     print(f"Output shape: {y.shape}")
