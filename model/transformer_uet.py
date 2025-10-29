

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np

# ----------------------------- 基础卷积块 -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐尺寸（可能存在 off-by-one）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ----------------------------- Transformer 模块 -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, max_len, d_model
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: B, N, C
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        # x: B, N, C
        x_attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x_attn
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        return x

class BottleneckTransformer(nn.Module):
    """在 bottleneck 处将 feature map 展平成 tokens，经过若干 transformer 层后再恢复回 feature map。"""
    def __init__(self, in_channels, token_dim, num_layers=2, patch_size=1, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.patch_size = patch_size  # 支持 1 代表每个像素作为 token
        self.proj = nn.Conv2d(in_channels, token_dim, kernel_size=1)
        self.pos = PositionalEncoding(token_dim, max_len=10000)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(token_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.unproj = nn.Conv2d(token_dim, in_channels, kernel_size=1)

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        x = self.proj(x)  # B, token_dim, H, W
        # Flatten spatial -> tokens
        x = x.flatten(2).transpose(1, 2)  # B, N(H*W), token_dim
        x = self.pos(x)
        for blk in self.transformer_blocks:
            x = blk(x)
        # Restore
        x = x.transpose(1, 2).view(B, self.token_dim, H, W)
        x = self.unproj(x)  # B, C, H, W  (恢复回 in_channels)
        return x

# ----------------------------- U-Net + Transformer -----------------------------
class UNetTransformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_c=32, bilinear=True,
                 transformer_dim=128, transformer_layers=2, transformer_heads=8):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor)

        # bottleneck transformer
        self.bottleneck = BottleneckTransformer(in_channels=base_c*16 // factor,
                                                token_dim=transformer_dim,
                                                num_layers=transformer_layers,
                                                num_heads=transformer_heads)

        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear)
        self.up4 = Up(base_c*2, base_c, bilinear)
        self.outc = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # transformer 在 x5 上操作
        x5 = self.bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        loss = None  # 占位符，保持接口一致
        return logits,loss
