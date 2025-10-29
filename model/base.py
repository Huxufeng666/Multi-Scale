import torch
import torch.nn as nn
import torch.nn.functional as F
from .CCA import CristCoistAttention, CCA_FusionModule
from utils.cos_loss import CosineSimLoss
from .Decouple import DecoupleLayer
from .Efficient_Features import EfficientFeature


# -----------------------------------------
# 🔹 基础卷积块
# -----------------------------------------
class BasicBlock(nn.Module):
    """
    一个基础的 ResNet 块，包含两个 3x3 卷积和一个残差连接。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接 (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果步长不为1（下采样）或通道数不同，则使用 1x1 卷积来匹配维度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x) # 记录 x 的“身份”

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加残差
        out += identity
        out = self.relu(out) # 最终激活
        
        return out

# -----------------------------------------
# 🔹 下采样模块
# -----------------------------------------
class Down(nn.Module):
    """
    下采样模块：使用 MaxPool2d + 一个 BasicBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_res = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(in_channels, out_channels) # 使用残差块替换 DoubleConv
        )

    def forward(self, x):
        return self.pool_res(x)
# -----------------------------------------
# 🔹 上采样模块
# -----------------------------------------
class Up(nn.Module):
    """
    上采样模块：使用 ConvTranspose2d + Concat + BasicBlock
    """
    def __init__(self, in_channels_after_cat, out_channels):
        super().__init__()
        
        # 1. 计算通道数
        self.in_channels_skip = out_channels
        self.in_channels_below = in_channels_after_cat - self.in_channels_skip
        self.up = nn.ConvTranspose2d(self.in_channels_below, self.in_channels_skip, kernel_size=2, stride=2)
     
        self.res_block = BasicBlock(2 * out_channels, out_channels)

    def forward(self, x_below, x_skip):
        x_up = self.up(x_below) # (例如: 1024 -> 512)
        
        # U-Net 中经常需要处理上采样导致的 1 像素差异
        if x_up.size() != x_skip.size():
            # 使用插值来强制对齐
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=True)
            
        # 2. 拼接 Skip Connection
        x = torch.cat([x_up, x_skip], dim=1) # (例如: 512 + 512 = 1024)
    
        return self.res_block(x)




# -----------------------------------------
# 🔹 输出卷积
# -----------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn + x


class DecoupleWithSA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.decouple = DecoupleLayer(in_ch)
        self.spatial_attn = SpatialAttention()
        self.mix_conv = nn.Conv2d(in_ch * 2, in_ch, 1)

    def forward(self, x, gt=None):
        F_map, B_map = self.decouple(x)
        out = torch.cat([F_map, B_map], dim=1)
        out = self.mix_conv(out)
        out = self.spatial_attn(out)
        return out


# -----------------------------------------
# 🔹 U-Net 主体结构
# -----------------------------------------
class UNet_EncoderLoss(nn.Module):
    def __init__(self, n_channels=1, base_ch=64, use_decouple=True):
        super(UNet_EncoderLoss, self).__init__()
        self.use_decouple = use_decouple

        self.inc = BasicBlock(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*16)

        # self.inc = CristCoistAttention(base_ch)
        self.cca1 = CristCoistAttention(base_ch * 2)
        self.cca2 = CristCoistAttention(base_ch * 4)
        self.cca3 = CristCoistAttention(base_ch * 8)
        self.cca4 = CristCoistAttention(base_ch * 16)
        
        
        self.fusion = CCA_FusionModule(
            [ base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16 ]
        )


        if use_decouple:
            self.decouple_sa = DecoupleWithSA(1024)

        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_channels)


        self.cosine_loss = CosineSimLoss(in_channels=base_ch * 8, proj_dim=256)

    def forward(self, x, gt=None):
        x1 = self.inc(x)         # (B,64,H, W)
        x2 = self.down1(x1)      # (B,128,H/2,W/2)
        x3 = self.down2(x2)      # (B,256,H/4,W/4)
        x4 = self.down3(x3)      # (B,512,H/8,W/8)
        x5 = self.down4(x4)      # (B,1024,H/16,W/16)
        
        cca1 = self.cca1(x2)
        cca2 = self.cca2(x3)
        cca3 = self.cca3(x4)
        cca4 = self.cca4(x5)

        cca_fusion = self.fusion([cca1, cca2, cca3, cca4])  # 融合后的特征 (B,512,H/8,W/8)
        
        
        if self.use_decouple:
            x5 = self.decouple_sa(x5, gt)  # (B,1024,H/16,W/16)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        
        loss_cos = None
        if self.training and gt is not None:
            # 将GT下采样到相同尺寸
            gt_down4 = F.interpolate(gt, size=x4.shape[2:], mode='bilinear', align_corners=False)
            gt_downfusion = F.interpolate(gt, size=cca_fusion.shape[2:], mode='bilinear', align_corners=False)

            loss_cos_block4 = self.cosine_loss(x4, gt_down4)
            loss_cos_fusion = self.cosine_loss(cca_fusion, gt_downfusion)

            # 综合相似度损失
            loss_cos = 0.5 * (loss_cos_block4 + loss_cos_fusion)
        
        
        
        return logits,loss_cos



# if __name__ == "__main__":
#     # 创建随机输入
#     x = torch.randn(2, 1, 256, 256)  # batch=2, 单通道, 256x256
    
#     # 创建模型（是否启用 decouple 模块）
#     model = UNet()
    
#     # 前向传播测试
#     with torch.no_grad():
#         y = model(x)
    
#         # print("✅ Forward success!")
            
#         # print(f"Input shape:  {x.shape}")
#         print(f"Output shape: {y.shape}")

'''
docker run -dit --name muitl-scale  --gpus all -v  /home/ami-1/HUXUFENG/Multi-scale/:/workspace -v /home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/:/workspace/dataset  pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
'''