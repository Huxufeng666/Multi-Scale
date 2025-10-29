

import torch.nn as nn
import torch.nn.functional as F

class CosineSimLoss(nn.Module):
    def __init__(self, in_channels, proj_dim=256):
        """
        in_channels: 输入特征的通道数 C
        proj_dim: 投影维度 D
        """
        super(CosineSimLoss, self).__init__()
        self.W_f = nn.Linear(in_channels, proj_dim)
        self.W_g = nn.Linear(1, proj_dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, F_feat, M_feat, G_mask):
        """
        F_feat, M_feat: [B, C, H, W]
        G_mask: [B, 1, H', W']
        """
        B, C, H, W = F_feat.shape

        # 1. 全局平均池化
        F_hat = F.adaptive_avg_pool2d(F_feat, 1).view(B, C)  # [B, C]
        M_hat = F.adaptive_avg_pool2d(M_feat, 1).view(B, C)  # [B, C]

        # G_mask -> avg_pool -> mean
        G_hat = F.adaptive_avg_pool2d(G_mask, 16).mean(dim=[1, 2, 3], keepdim=True)  # [B, 1]

        # 2. 线性映射
        f = self.W_f(F_hat)  # [B, D]
        m = self.W_f(M_hat)  # [B, D]
        g = self.W_g(G_hat)  # [B, D]

        # 3. 余弦相似度
        loss = (1 - self.cos(f, g)).mean() + (1 - self.cos(m, g)).mean()
        return loss
